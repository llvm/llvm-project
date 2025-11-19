// Ensure we support the various CPU architecture names.
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=nocona 2>&1 \
// RUN:   | FileCheck %s -check-prefix=nocona
// nocona: "-target-cpu" "nocona"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=core2 2>&1 \
// RUN:   | FileCheck %s -check-prefix=core2
// core2: "-target-cpu" "core2"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=penryn 2>&1 \
// RUN:   | FileCheck %s -check-prefix=penryn
// penryn: "-target-cpu" "penryn"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=nehalem 2>&1 \
// RUN:   | FileCheck %s -check-prefix=nehalem
// nehalem: "-target-cpu" "nehalem"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=westmere 2>&1 \
// RUN:   | FileCheck %s -check-prefix=westmere
// westmere: "-target-cpu" "westmere"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=sandybridge 2>&1 \
// RUN:   | FileCheck %s -check-prefix=sandybridge
// sandybridge: "-target-cpu" "sandybridge"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=ivybridge 2>&1 \
// RUN:   | FileCheck %s -check-prefix=ivybridge
// ivybridge: "-target-cpu" "ivybridge"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=haswell 2>&1 \
// RUN:   | FileCheck %s -check-prefix=haswell
// haswell: "-target-cpu" "haswell"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=broadwell 2>&1 \
// RUN:   | FileCheck %s -check-prefix=broadwell
// broadwell: "-target-cpu" "broadwell"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=skylake 2>&1 \
// RUN:   | FileCheck %s -check-prefix=skylake
// skylake: "-target-cpu" "skylake"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=skylake-avx512 2>&1 \
// RUN:   | FileCheck %s -check-prefix=skylake-avx512
// skylake-avx512: "-target-cpu" "skylake-avx512"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=skx 2>&1 \
// RUN:   | FileCheck %s -check-prefix=skx
// skx: "-target-cpu" "skx"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=cascadelake 2>&1 \
// RUN:   | FileCheck %s -check-prefix=cascadelake
// cascadelake: "-target-cpu" "cascadelake"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=cooperlake 2>&1 \
// RUN:   | FileCheck %s -check-prefix=cooperlake
// cooperlake: "-target-cpu" "cooperlake"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=knl 2>&1 \
// RUN:   | FileCheck %s -check-prefix=knl
// knl: "-target-cpu" "knl"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=knm 2>&1 \
// RUN:   | FileCheck %s -check-prefix=knm
// knm: "-target-cpu" "knm"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=cannonlake 2>&1 \
// RUN:   | FileCheck %s -check-prefix=cannonlake
// cannonlake: "-target-cpu" "cannonlake"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=icelake-client 2>&1 \
// RUN:   | FileCheck %s -check-prefix=icelake-client
// icelake-client: "-target-cpu" "icelake-client"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=rocketlake 2>&1 \
// RUN:   | FileCheck %s -check-prefix=rocketlake
// rocketlake: "-target-cpu" "rocketlake"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=icelake-server 2>&1 \
// RUN:   | FileCheck %s -check-prefix=icelake-server
// icelake-server: "-target-cpu" "icelake-server"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=tigerlake 2>&1 \
// RUN:   | FileCheck %s -check-prefix=tigerlake
// tigerlake: "-target-cpu" "tigerlake"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=alderlake 2>&1 \
// RUN:   | FileCheck %s -check-prefix=alderlake
// alderlake: "-target-cpu" "alderlake"
//
// RUN: %clang --target=x86_64 -c -### %s -march=raptorlake 2>&1 \
// RUN:   | FileCheck %s -check-prefix=raptorlake
// raptorlake: "-target-cpu" "raptorlake"
//
// RUN: %clang --target=x86_64 -c -### %s -march=meteorlake 2>&1 \
// RUN:   | FileCheck %s -check-prefix=meteorlake
// meteorlake: "-target-cpu" "meteorlake"
//
// RUN: %clang --target=x86_64 -c -### %s -march=arrowlake 2>&1 \
// RUN:   | FileCheck %s -check-prefix=arrowlake
// arrowlake: "-target-cpu" "arrowlake"
//
// RUN: %clang --target=x86_64 -c -### %s -march=arrowlake-s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=arrowlake-s
// arrowlake-s: "-target-cpu" "arrowlake-s"
//
// RUN: %clang --target=x86_64 -c -### %s -march=lunarlake 2>&1 \
// RUN:   | FileCheck %s -check-prefix=lunarlake
// lunarlake: "-target-cpu" "lunarlake"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=gracemont 2>&1 \
// RUN:   | FileCheck %s -check-prefix=gracemont
// gracemont: "-target-cpu" "gracemont"
//
// RUN: %clang --target=x86_64 -c -### %s -march=pantherlake 2>&1 \
// RUN:   | FileCheck %s -check-prefix=pantherlake
// pantherlake: "-target-cpu" "pantherlake"
//
// RUN: %clang --target=x86_64 -c -### %s -march=wildcatlake 2>&1 \
// RUN:   | FileCheck %s -check-prefix=wildcatlake
// wildcatlake: "-target-cpu" "wildcatlake"
//
// RUN: %clang --target=x86_64 -c -### %s -march=novalake 2>&1 \
// RUN:   | FileCheck %s -check-prefix=novalake
// novalake: "-target-cpu" "novalake"
//
// RUN: %clang --target=x86_64 -c -### %s -march=clearwaterforest 2>&1 \
// RUN:   | FileCheck %s -check-prefix=clearwaterforest
// clearwaterforest: "-target-cpu" "clearwaterforest"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=diamondrapids 2>&1 \
// RUN:   | FileCheck %s -check-prefix=diamondrapids
// diamondrapids: "-target-cpu" "diamondrapids"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=lakemont 2>&1 \
// RUN:   | FileCheck %s -check-prefix=lakemont
// lakemont: "-target-cpu" "lakemont"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=bonnell 2>&1 \
// RUN:   | FileCheck %s -check-prefix=bonnell
// bonnell: "-target-cpu" "bonnell"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=silvermont 2>&1 \
// RUN:   | FileCheck %s -check-prefix=silvermont
// silvermont: "-target-cpu" "silvermont"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=goldmont 2>&1 \
// RUN:   | FileCheck %s -check-prefix=goldmont
// goldmont: "-target-cpu" "goldmont"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=goldmont-plus 2>&1 \
// RUN:   | FileCheck %s -check-prefix=goldmont-plus
// goldmont-plus: "-target-cpu" "goldmont-plus"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=tremont 2>&1 \
// RUN:   | FileCheck %s -check-prefix=tremont
// tremont: "-target-cpu" "tremont"
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=sierraforest 2>&1 \
// RUN:   | FileCheck %s -check-prefix=sierraforest
// sierraforest: "-target-cpu" "sierraforest"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=grandridge 2>&1 \
// RUN:   | FileCheck %s -check-prefix=grandridge
// grandridge: "-target-cpu" "grandridge"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=graniterapids 2>&1 \
// RUN:   | FileCheck %s -check-prefix=graniterapids
// graniterapids: "-target-cpu" "graniterapids"
//
// RUN: %clang --target=x86_64-unknown-unknown -c -### %s -march=graniterapids-d 2>&1 \
// RUN:   | FileCheck %s -check-prefix=graniterapids-d
// graniterapids-d: "-target-cpu" "graniterapids-d"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=emeraldrapids 2>&1 \
// RUN:   | FileCheck %s -check-prefix=emeraldrapids
// emeraldrapids: "-target-cpu" "emeraldrapids"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=sapphirerapids 2>&1 \
// RUN:   | FileCheck %s -check-prefix=sapphirerapids
// sapphirerapids: "-target-cpu" "sapphirerapids"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=k8 2>&1 \
// RUN:   | FileCheck %s -check-prefix=k8
// k8: "-target-cpu" "k8"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=opteron 2>&1 \
// RUN:   | FileCheck %s -check-prefix=opteron
// opteron: "-target-cpu" "opteron"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=athlon64 2>&1 \
// RUN:   | FileCheck %s -check-prefix=athlon64
// athlon64: "-target-cpu" "athlon64"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=athlon-fx 2>&1 \
// RUN:   | FileCheck %s -check-prefix=athlon-fx
// athlon-fx: "-target-cpu" "athlon-fx"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=k8-sse3 2>&1 \
// RUN:   | FileCheck %s -check-prefix=k8-sse3
// k8-sse3: "-target-cpu" "k8-sse3"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=opteron-sse3 2>&1 \
// RUN:   | FileCheck %s -check-prefix=opteron-sse3
// opteron-sse3: "-target-cpu" "opteron-sse3"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=athlon64-sse3 2>&1 \
// RUN:   | FileCheck %s -check-prefix=athlon64-sse3
// athlon64-sse3: "-target-cpu" "athlon64-sse3"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=amdfam10 2>&1 \
// RUN:   | FileCheck %s -check-prefix=amdfam10
// amdfam10: "-target-cpu" "amdfam10"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=barcelona 2>&1 \
// RUN:   | FileCheck %s -check-prefix=barcelona
// barcelona: "-target-cpu" "barcelona"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=bdver1 2>&1 \
// RUN:   | FileCheck %s -check-prefix=bdver1
// bdver1: "-target-cpu" "bdver1"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=bdver2 2>&1 \
// RUN:   | FileCheck %s -check-prefix=bdver2
// bdver2: "-target-cpu" "bdver2"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=bdver3 2>&1 \
// RUN:   | FileCheck %s -check-prefix=bdver3
// bdver3: "-target-cpu" "bdver3"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=bdver4 2>&1 \
// RUN:   | FileCheck %s -check-prefix=bdver4
// bdver4: "-target-cpu" "bdver4"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=btver1 2>&1 \
// RUN:   | FileCheck %s -check-prefix=btver1
// btver1: "-target-cpu" "btver1"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=btver2 2>&1 \
// RUN:   | FileCheck %s -check-prefix=btver2
// btver2: "-target-cpu" "btver2"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=znver1 2>&1 \
// RUN:   | FileCheck %s -check-prefix=znver1
// znver1: "-target-cpu" "znver1"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=znver2 2>&1 \
// RUN:   | FileCheck %s -check-prefix=znver2
// znver2: "-target-cpu" "znver2"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=znver3 2>&1 \
// RUN:   | FileCheck %s -check-prefix=znver3
// znver3: "-target-cpu" "znver3"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=znver4 2>&1 \
// RUN:   | FileCheck %s -check-prefix=znver4
// znver4: "-target-cpu" "znver4"
//
// RUN: %clang -target x86_64-unknown-unknown -c -### %s -march=znver5 2>&1 \
// RUN:   | FileCheck %s -check-prefix=znver5
// znver5: "-target-cpu" "znver5"

// RUN: %clang -target x86_64 -c -### %s -march=x86-64 2>&1 | FileCheck %s --check-prefix=x86-64
// x86-64: "-target-cpu" "x86-64"
// RUN: %clang -target x86_64 -c -### %s -march=x86-64-v2 2>&1 | FileCheck %s --check-prefix=x86-64-v2
// x86-64-v2: "-target-cpu" "x86-64-v2"
// RUN: %clang -target x86_64 -c -### %s -march=x86-64-v3 2>&1 | FileCheck %s --check-prefix=x86-64-v3
// x86-64-v3: "-target-cpu" "x86-64-v3"
// RUN: %clang -target x86_64 -c -### %s -march=x86-64-v4 2>&1 | FileCheck %s --check-prefix=x86-64-v4
// x86-64-v4: "-target-cpu" "x86-64-v4"
