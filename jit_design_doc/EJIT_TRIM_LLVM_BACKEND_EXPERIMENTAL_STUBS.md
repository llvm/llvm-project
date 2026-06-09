# EJIT 裸核桩函数设计文档

**版本**: 1.0
**日期**: 2026-05-26
**关联**: EJIT_LIBRARY_TRIMMING.md
**目标**: 梳理裸核环境下 EJIT 二进制依赖的全部外部符号，分类给出桩实现方案

---

## 1. 分析背景

以 `-Os -ffunction-sections -fdata-sections` 构建 EJIT，通过 `readelf -sW` 分析最终二进制（nostrip）的未定义符号，梳理裸核需实现的外部依赖。

## 2. 符号分类总览

| 类别 | 数量 | 来源 | 裸核方案 |
|---|---|---|---|
| C 标准库 | ~80 | `memcpy`, `malloc`, `printf` 等 | **newlib / picolibc** 提供 |
| C++ 标准库 | ~60 | `std::string`, `std::map`, `operator new` 等 | **libstdc++**（hosted 模式） |
| pthread | 8 | `pthread_mutex_*`, `pthread_rwlock_*` | **桩函数**，全部返回 0 |
| OS 内存 | 3 | `mmap`, `munmap`, `mprotect` | **slab 分配器**（运行）；桩函数（链接） |
| 动态加载 | 4 | `dlopen`, `dlsym`, `dlclose`, `dlerror` | **桩函数**，返回 NULL/-1 |
| 文件 I/O | ~20 | `open`, `close`, `read`, `write`, `stat` 等 | **桩函数**，返回 -1 |
| 进程管理 | ~10 | `getpid`, `getenv`, `fork`, `execve` 等 | **桩函数**，返回常量/错误 |
| 信号处理 | 8 | `sigaction`, `kill`, `raise` 等 | **桩函数**，全部 no-op |
| 时间 | 2 | `time`, `clock_gettime` | **桩函数**，返回 0 |
| 编译器内置 | 少量 | `__register_frame`, `__morestack` 等 | 工具链自动处理 |

## 3. 各类符号详细分析

### 3.1 pthread

**来源**: `std::mutex` 底层实现 → `pthread_mutex_lock/unlock`。OrcJIT/Support/JITLink 共 ~190 处 `std::mutex` 引用。

裸核单线程运行，全部可 no-op。

```c
int pthread_mutex_lock(void *m)          { return 0; }
int pthread_mutex_unlock(void *m)        { return 0; }
int pthread_mutex_init(void *m, const void *a) { return 0; }
int pthread_mutex_destroy(void *m)       { return 0; }
int pthread_rwlock_rdlock(void *r)       { return 0; }
int pthread_rwlock_wrlock(void *r)       { return 0; }
int pthread_rwlock_unlock(void *r)       { return 0; }
int pthread_once(int *o, void (*f)(void)){ if(!*o){*o=1;f();} return 0; }
int pthread_sigmask(int h, const void *s, void *o) { return 0; }
```

**注意**: 裸核工具链若**不提供 `<mutex>` 头文件**，需要启用 `-DEJIT_TRIM_LLVM_BACKEND_EXPERIMENTAL_MUTEX=ON`，CMake 会通过 `target_include_directories SYSTEM BEFORE` 注入 `BareMetal/mutex` 等 no-op 头文件。若工具链提供 `<mutex>`（hosted 模式），只需 pthread 桩函数即可。

### 3.2 OS 内存

**来源**: `JITLink InProcessMemoryManager` → `sys::Memory::allocateMappedMemory()` → `mmap()`。

```c
// 桩函数 — 仅让链接通过。运行时需 slab 分配器替代。
void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
    errno = ENOMEM;
    return MAP_FAILED;
}
int munmap(void *addr, size_t len) { return 0; }
int mprotect(void *addr, size_t len, int prot) { return 0; }
```

> **关键**: `mmap` 桩只让链接通过。裸核 JIT 编译要能执行，必须在 `EJitOrcEngine::Create()` 中注入自定义 `JITLinkMemoryManager`（基于静态预分配 slab）。`mprotect` 在裸核下通常是 no-op（内存默认 RWX）。

### 3.3 动态加载

**来源**: `EPCDynamicLibrarySearchGenerator` → `sys::DynamicLibrary::getPermanentLibrary()` → `dlopen()`。LLJIT 默认创建此生成器以解析宿主进程符号。

裸核下 `setLinkProcessSymbolsByDefault(false)` + `ejit_register_symbol()` 替代，不经过此路径。

```c
void *dlopen(const char *f, int m)  { return NULL; }
void *dlsym(void *h, const char *s) { return NULL; }
int   dlclose(void *h)              { return -1; }
char *dlerror(void)                 { return "bare-metal: no dynamic libs"; }
```

### 3.4 文件 I/O

**来源**: `Support/raw_ostream.cpp` (raw_fd_ostream)、`Support/MemoryBuffer.cpp` (MemoryBuffer::getFile)、`Support/Path.cpp` (sys::fs 系列)。

裸核 JIT 只用 `MemoryBuffer::getMemBuffer()`（内存 buffer），不调文件路径。全部可返回错误。

```c
int open(const char *p, int f, ...)  { errno = ENOENT; return -1; }
int close(int fd)                    { return 0; }
ssize_t read(int fd, void *b, size_t n) { errno = EBADF; return -1; }
ssize_t write(int fd, const void *b, size_t n) { errno = EBADF; return -1; }
int fstat(int fd, void *s)           { errno = EBADF; return -1; }
int stat(const char *p, void *s)     { errno = ENOENT; return -1; }
off_t lseek(int fd, off_t o, int w)  { errno = EBADF; return -1; }
int fcntl(int fd, int c, ...)        { errno = EBADF; return -1; }
```

### 3.5 进程管理

**来源**: `Support/Unix/Process.inc`、`Support/Unix/Program.inc`。

```c
pid_t getpid(void)                { return 1; }
char *getenv(const char *n)       { return NULL; }
long  sysconf(int name)           { return (name == _SC_PAGE_SIZE) ? 4096 : -1; }
pid_t fork(void)                  { errno = ENOSYS; return -1; }
int   execve(const char *p, char *const a[], char *const e[]) { errno = ENOSYS; return -1; }
pid_t waitpid(pid_t p, int *s, int o) { errno = ECHILD; return -1; }
// posix_spawn
int posix_spawn(pid_t *p, const char *f, const void *fa,
                const void *attr, char *const a[], char *const e[]) { return ENOSYS; }
int posix_spawn_file_actions_init(void *fa)    { return 0; }
int posix_spawn_file_actions_destroy(void *fa) { return 0; }
int posix_spawn_file_actions_addopen(void *fa, int fd, const char *p, int f, mode_t m) { return 0; }
int posix_spawn_file_actions_adddup2(void *fa, int fd, int nfd) { return 0; }
```

> `sysconf(_SC_PAGE_SIZE)` 是唯一 JIT 实际需要值的（`MemoryMapper` 做页对齐），返回 4096 即可。

### 3.6 信号处理

**来源**: `Support/Unix/Signals.inc` — LLVM 崩溃处理/栈回溯基础设施。JIT 核心不碰。

```c
int sigaction(int s, const void *a, void *o) { return 0; }
int sigprocmask(int h, const void *s, void *o) { return 0; }
int sigemptyset(void *s)  { return 0; }
int sigfillset(void *s)   { return 0; }
int sigaltstack(const void *s, void *o) { return 0; }
int kill(pid_t p, int s)  { return 0; }
int raise(int s)          { return 0; }
```

### 3.7 时间

**来源**: libstdc++ `<chrono>` 实现 → `clock_gettime()`。Support/Unix/Process.inc → `::times()`。

```c
time_t time(time_t *t) { if(t) *t=0; return 0; }
int clock_gettime(clockid_t c, struct timespec *ts) { ts->tv_sec=0; ts->tv_nsec=0; return 0; }
int gettimeofday(struct timeval *tv, void *tz) { tv->tv_sec=0; tv->tv_usec=0; return 0; }
```

## 4. 桩函数库结构

```
ejit_test/bare-metal/
├── libpthread_stub.c       ← pthread 桩 (8 个函数)
├── libmem_stub.c           ← mmap/munmap/mprotect (3 个函数)
├── libdl_stub.c            ← dlopen/dlsym/dlclose/dlerror (4 个函数)
├── libfs_stub.c            ← open/close/read/write/stat/fstat/lseek/fcntl (~20 个函数)
├── libproc_stub.c          ← getpid/getenv/sysconf/fork/execve/wait/posix_spawn (~15 个函数)
├── libsignal_stub.c        ← sigaction/kill/raise 等 (8 个函数)
├── libtime_stub.c          ← time/clock_gettime/gettimeofday (3 个函数)
└── libbaremetal.a          ← 全部桩函数合并产物
```

## 5. 裸核链接命令

```bash
aarch64-none-elf-g++ \
  -Os -Wl,--gc-sections -Wl,--strip-all \
  ejit_test/lipo/ejit.o \           # EJIT + LLVM 合并产物 (37 MB)
  user_app.o \                       # 用户代码
  libbaremetal.a \                   # 桩函数库 (~5 KB)
  -lstdc++ -lm -lc -lgcc \           # hosted 工具链自带
  -T bare_metal.ld \                 # 裸核链接脚本
  -o firmware.elf
```

## 6. 遗留工作

| 项目 | 优先级 | 说明 |
|---|---|---|
| `JITLinkMemoryManager` slab 实现 | **高** | `mmap` 桩只让链接通过，运行时需静态 slab 分配器替代 |
| `BareMetal/` 头文件验证 | 中 | 需在裸核工具链上确认 `<mutex>` 是否存在 |
| 桩函数库编译 | 低 | 60 行 C 代码，随时可写 |
| `EJIT_TRIM_LLVM_BACKEND_EXPERIMENTAL_MUTEX` 测试 | 低 | 裸核工具链到位后验证 |

---

*文档版本: 1.0*
*创建日期: 2026-05-26*
