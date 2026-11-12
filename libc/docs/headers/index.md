# Implementation Status

Purely compiler-provided freestanding C headers (`iso646.h`, `stdalign.h`,
`stdarg.h`, `stdatomic.h`, `stdbool.h`, `stddef.h`, `stdnoreturn.h`, and
`tgmath.h`) come from the host compiler (`Clang` or `GCC`) rather than
LLVM-libc and are omitted below.

```{toctree}
:maxdepth: 1

aio
arpa/inet
assert
byteswap
complex
cpio
ctype
dirent
dlfcn
endian
errno
fcntl
fenv
float
fmtmsg
fnmatch
ftw
glob
grp
iconv
inttypes
langinfo
libgen
limits
locale
math
monetary
mqueue
ndbm
net/if
netdb
netinet/in
netinet/tcp
nl_types
poll
pwd
regex
sched
search
semaphore
setjmp
signal
spawn
stdbit
stdckdint
stdfix
stdint
stdio
stdlib
string
strings
sys/ipc
sys/mman
sys/msg
sys/resource
sys/select
sys/sem
sys/shm
sys/socket
sys/stat
sys/statvfs
sys/time
sys/times
sys/types
sys/uio
sys/un
sys/utsname
sys/wait
syslog
tar
termios
threads
time
uchar
ulimit
unistd
utime
utmpx
wchar
wctype
pthread
wordexp
```

