/**
 * EJIT 优化等级测试 — 覆盖 L1 / L2 / L3
 *
 * 每个等级独立进程 (EJitRegistrationStore 只能 consume 一次),
 * 验证 JIT 编译成功 (entries > 0) 且结果正确。
 *
 * 用法:
 *   ./ejit_opt_level L1    # 测试 L1 (SCCP + ADCE + SimplifyCFG)
 *   ./ejit_opt_level L2    # 测试 L2 (L1 + AlwaysInliner + SimplifyCFG)
 *   ./ejit_opt_level L3    # 测试 L3 (L2 + LoopSimplify + LoopFullUnroll + Promote)
 *
 * 编译:
 *   build/bin/clang -O2 -c ejit_test/ejit_opt_level_test.c -o /tmp/opt_level.o
 * 链接:
 *   LIBS=$(ls build_x86/lib/x.a)
 *   clang++ -Os -Wl,--gc-sections /tmp/opt_level.o \
 *     -Wl,--whole-archive $LIBS -Wl,--no-whole-archive \
 *     -lpthread -ldl -o /tmp/ejit_opt_level
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//===-- 外部 API ------------------------------------------------------------===//

typedef enum { EJIT_OK = 0 } ejit_status_t;
typedef enum { EJIT_COMPILE_SYNC = 0 } ejit_compile_mode_t;
typedef enum { EJIT_OPT_L1 = 1, EJIT_OPT_L2 = 2, EJIT_OPT_L3 = 3 } ejit_opt_level_t;

typedef struct {
  ejit_compile_mode_t compileMode;
  ejit_opt_level_t optLevel;
  size_t maxCodeMemory, maxDataMemory, maxCacheEntries, maxCacheSize;
  bool enableLogger;
  const char *dumpJITDir;
} ejit_config_t;

typedef struct {
  size_t entryCount, totalCodeSize, maxSize;
  unsigned long long hits, misses, evictions;
} ejit_stats_t;

extern ejit_status_t ejit_init(const ejit_config_t *cfg);
extern void          ejit_shutdown(void);
extern ejit_status_t ejit_activate(const char *name, unsigned char idx);
extern ejit_status_t ejit_get_stats(ejit_stats_t *s);

//===-- 数据结构 -------------------------------------------------------------===//

struct DevInfo {
    __attribute__((ejit_may_const)) uint32_t type;      // 1=BBU, 2=RRU, 3=AAU
    __attribute__((ejit_may_const)) uint32_t maxPower;
    __attribute__((ejit_may_const)) uint32_t channels;
    uint32_t uptime;
};

struct CarrCfg {
    __attribute__((ejit_may_const)) uint32_t freqBand;
    __attribute__((ejit_may_const)) uint32_t bandwidth;
    __attribute__((ejit_may_const)) bool     mimo;
    uint32_t txPower;
};

//===-- 全局变量 -------------------------------------------------------------===//

__attribute__((ejit_period("static"))) struct DevInfo g_dev;

#define N_CARR 8
__attribute__((ejit_period_arr("carrier"))) struct CarrCfg g_carr[N_CARR];

//===-- always_inline helper (L2 内敛目标) ----------------------------------===//

__attribute__((always_inline))
static inline uint32_t power_offset(uint32_t base, uint32_t nchan) {
    return base + nchan * 3;
}

__attribute__((always_inline))
static inline uint32_t mimo_gain(uint32_t pwr, bool mimo) {
    return mimo ? pwr + 6 : pwr;
}

//===-- ejit_entry ----------------------------------------------------------===//

__attribute__((ejit_entry))
uint32_t process_carr(
    __attribute__((ejit_period_arr_ind("carrier"))) uint8_t ci)
{
    uint32_t r = 0;

    // L1: device type switch → constant-folded
    switch (g_dev.type) {
    case 1: r += 1000 + g_dev.maxPower*2 + g_dev.channels*5; break;
    case 2: r += 2000 + g_dev.maxPower*3; break;
    case 3: r += 3000 + g_dev.maxPower*4; break;
    default: r += 500;
    }

    // L1: carrier fields substituted → constant-folded branches
    uint32_t fb  = g_carr[ci].freqBand;
    uint32_t bw  = g_carr[ci].bandwidth;
    bool     mi  = g_carr[ci].mimo;

    switch (fb) { case 78: r+=78; break; case 41: r+=41; break;
                  case 28: r+=28; break; default: r+=100; }
    if      (bw==100) r+=10; else if (bw==40) r+=40;
    else if (bw==20)  r+=20; else             r+=1;
    if (mi) r += 6;

    // L2: always_inline inlined → constants propagated through
    uint32_t p = power_offset(40, g_dev.channels);
    p = mimo_gain(p, mi);
    r += p;

    return r;
}

//===-- main ----------------------------------------------------------------===//

int main(int argc, char **argv)
{
    const char *ls = (argc>1) ? argv[1] : "L1";
    ejit_opt_level_t lv;
    const char *ln;
    if      (!strcmp(ls,"L3")||!strcmp(ls,"l3")) { lv=EJIT_OPT_L3; ln="L3"; }
    else if (!strcmp(ls,"L2")||!strcmp(ls,"l2")) { lv=EJIT_OPT_L2; ln="L2"; }
    else                                         { lv=EJIT_OPT_L1; ln="L1"; }

    printf("=== EJIT Opt Level: %s ===\n\n", ln);

    // Init globals: type=3(AAU), maxPower=120, channels=4
    g_dev.type=3; g_dev.maxPower=120; g_dev.channels=4;
    for (int i=0; i<N_CARR; i++) {
        g_carr[i].freqBand  = (i%3==0)?78:((i%3==1)?41:28);
        g_carr[i].bandwidth = 100;
        g_carr[i].mimo      = (i%2==0);
    }

    // Init EJIT
    ejit_config_t c; memset(&c,0,sizeof(c));
    c.compileMode=EJIT_COMPILE_SYNC; c.optLevel=lv;
    c.maxCodeMemory=512*1024; c.maxDataMemory=256*1024;
    c.maxCacheEntries=64; c.maxCacheSize=1024*1024;
    if (ejit_init(&c)!=EJIT_OK) { printf("FAIL: ejit_init\n"); return 1; }
    ejit_activate("carrier",0);
    ejit_activate("carrier",1);
    ejit_activate("carrier",2);

    int failures = 0;

    // Run with 3 different carrierIdx values
    for (int t=0; t<3; t++) {
        uint8_t ci = (uint8_t)t;
        uint32_t res = process_carr(ci);

        // Compute expected (AOT ground truth — accesses real globals)
        uint32_t exp = 3480;  // AAU: 3000 + 120*4
        if      (ci%3==0) exp+=78;
        else if (ci%3==1) exp+=41;
        else             exp+=28;
        exp += 10;             // bw=100
        if (ci%2==0) exp += 6; // MIMO on for even ci
        // power: 40+4*3=52, +6 if mimo
        exp += 52 + ((ci%2==0)?6:0);

        ejit_stats_t s; memset(&s,0,sizeof(s)); ejit_get_stats(&s);
        int isJit = (s.entryCount > (size_t)t) ? 1 : 0;

        printf("[Test %d] ci=%u %s  result=%u expected=%u  ",
               t+1, ci, isJit?"[JIT]":"[AOT]", res, exp);
        if (res==exp) printf("[MATCH]\n");
        else { printf("[MISMATCH]\n"); failures++; }
    }

    ejit_stats_t sf; memset(&sf,0,sizeof(sf)); ejit_get_stats(&sf);
    printf("\nJIT: entries=%zu misses=%llu  ", sf.entryCount, sf.misses);
    if (sf.entryCount > 0) printf("[ACTIVE]\n");
    else { printf("[NOT ACTIVE]\n"); failures++; }

    ejit_shutdown();
    printf("\n=== %s: %d failures ===\n", ln, failures);
    return failures>0 ? 1 : 0;
}
