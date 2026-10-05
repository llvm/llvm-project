// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t-O0.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t-O0.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -disable-llvm-passes -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-CIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -disable-llvm-passes -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s

struct Str {
  char c;
  Str(const Str &);
  ~Str();
};

struct Io {
  virtual char *stream(const Str &, Str);
};

struct Replicas {
  virtual ~Replicas();
};

struct Proxy : Replicas, Io {
  ~Proxy();
};

struct Namd : Proxy {
  char *stream(const Str &, Str) override;
};

// The top-level const is written only on the definition, which is what marks
// the parameter's spill slot const.
char *Namd::stream(const Str &, const Str) { return nullptr; }

// The by-value parameter is forwarded without a copy, from behind a reference
// parameter and alongside a returned pointer.

// CIR-LABEL: cir.func{{.*}} @_ZThn8_N4Namd6streamERK3StrS0_
// CIR-SAME:    %{{[^ :]+}}: !cir.ptr<!rec_Namd> {llvm.noundef}
// CIR-SAME:    %{{[^ :]+}}: !cir.ptr<!rec_Str> {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}
// CIR-SAME:    %[[STR:[^ :]+]]: !cir.ptr<!rec_Str> {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nofreeobj, llvm.noundef}
// CIR:         cir.call @_ZN4Namd6streamERK3StrS0_(%{{[^,)]+}}, %{{[^,)]+}}, %[[STR]]) : (!cir.ptr<!rec_Namd> {llvm.align = 8 : i64, llvm.dereferenceable = 16 : i64, llvm.nonnull, llvm.noundef}, !cir.ptr<!rec_Str> {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, !cir.ptr<!rec_Str> {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nofreeobj, llvm.noundef}) -> (!cir.ptr<!s8i> {llvm.noundef})

// The overrider takes the same pointer without byval, which is the definition
// side of the same contract.

// LLVM-LABEL: define dso_local noundef ptr @_ZN4Namd6streamERK3StrS0_(
// LLVM-SAME:    ptr noundef nonnull align 8 dereferenceable(16) %{{[^,]+}},
// LLVM-SAME:    ptr noundef nonnull align 1 dereferenceable(1) %{{[^,]+}},
// LLVM-SAME:    ptr nofreeobj noundef align 1 dereferenceable(1) %{{[^,)]+}})

// LLVM-LABEL: define dso_local noundef ptr @_ZThn8_N4Namd6streamERK3StrS0_(
// LLVM-SAME:    ptr noundef %{{[^,]+}},
// LLVM-SAME:    ptr noundef nonnull align 1 dereferenceable(1) %{{[^,]+}},
// LLVM-SAME:    ptr nofreeobj noundef align 1 dereferenceable(1) %[[STR:[^,)]+]])
// LLVM-CIR:     call noundef ptr @_ZN4Namd6streamERK3StrS0_(ptr noundef nonnull align 8 dereferenceable(16) %{{[^,)]+}}, ptr noundef nonnull align 1 dereferenceable(1) %{{[^,)]+}}, ptr nofreeobj noundef align 1 dereferenceable(1) %[[STR]])
// OGCG:         tail call noundef ptr @_ZN4Namd6streamERK3StrS0_(ptr noundef nonnull align 8 dereferenceable(16) %{{[^,)]+}}, ptr noundef nonnull align 1 dereferenceable(1) %{{[^,)]+}}, ptr nofreeobj noundef align 1 dereferenceable(1) %[[STR]])
