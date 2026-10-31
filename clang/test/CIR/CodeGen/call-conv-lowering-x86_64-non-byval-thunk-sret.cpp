// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -disable-llvm-passes -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-CIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -disable-llvm-passes -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s

// Impl's key function is not defined here, so the vtable and its thunks are
// emitted only when optimizing.

struct Str {
  ~Str();
};

struct BBox {
  float a, b, c, d, e;
};

struct Base {
  virtual void anchor();
};

struct Renderer {
  virtual void textWithBBox(float, float, Str, unsigned);
  virtual BBox bbox(Str, unsigned);
};

struct Impl : Base, Renderer {
  void textWithBBox(float, float, const Str, unsigned);
  BBox bbox(const Str, unsigned);
};

void emit() { new Impl; }

// The by-value parameter sits behind two floats, so it is not argument zero.

// CIR-LABEL: cir.func{{.*}} @_ZThn8_N4Impl12textWithBBoxEff3Strj
// CIR-SAME:    %{{[^ :]+}}: !cir.ptr<!rec_Impl> {llvm.noundef}
// CIR-SAME:    %{{[^ :]+}}: !cir.float {llvm.noundef}
// CIR-SAME:    %{{[^ :]+}}: !cir.float {llvm.noundef}
// CIR-SAME:    %[[STR:[^ :]+]]: !cir.ptr<!rec_Str> {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nofreeobj, llvm.noundef}
// CIR-NOT:     cir.alloca {{.*}}!cir.ptr<!rec_Str>
// CIR:         cir.call @_ZN4Impl12textWithBBoxEff3Strj(%{{[^,)]+}}, %{{[^,)]+}}, %{{[^,)]+}}, %[[STR]], %{{[^,)]+}})

// An sret return prepends a block argument, so the parameter is not at the
// index its classification sits at once the signature is rewritten.

// CIR-LABEL: cir.func{{.*}} @_ZThn8_N4Impl4bboxE3Strj
// CIR-SAME:    llvm.sret = !rec_BBox
// CIR-SAME:    %{{[^ :]+}}: !cir.ptr<!rec_Impl> {llvm.noundef}
// CIR-SAME:    %[[SSTR:[^ :]+]]: !cir.ptr<!rec_Str> {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nofreeobj, llvm.noundef}
// CIR-NOT:     cir.alloca {{.*}}!cir.ptr<!rec_Str>
// CIR:         cir.call @_ZN4Impl4bboxE3Strj(%{{[^,)]+}}, %{{[^,)]+}}, %[[SSTR]], %{{[^,)]+}})

// LLVM-LABEL: define available_externally void @_ZThn8_N4Impl12textWithBBoxEff3Strj(
// LLVM-SAME:    ptr noundef %{{[^,]+}},
// LLVM-SAME:    float noundef %{{[^,]+}},
// LLVM-SAME:    float noundef %{{[^,]+}},
// LLVM-SAME:    ptr nofreeobj noundef align 1 dereferenceable(1) %[[LSTR:[^,]+]],
// LLVM-SAME:    i32 noundef %{{[^,)]+}})
// LLVM-CIR:     call void @_ZN4Impl12textWithBBoxEff3Strj(ptr noundef nonnull align 8 dereferenceable(16) %{{[^,)]+}}, float noundef %{{[^,)]+}}, float noundef %{{[^,)]+}}, ptr nofreeobj noundef align 1 dereferenceable(1) %[[LSTR]], i32 noundef %{{[^,)]+}})
// OGCG:         tail call void @_ZN4Impl12textWithBBoxEff3Strj(ptr noundef nonnull align 8 dereferenceable(16) %{{[^,)]+}}, float noundef %{{[^,)]+}}, float noundef %{{[^,)]+}}, ptr nofreeobj noundef align 1 dereferenceable(1) %[[LSTR]], i32 noundef %{{[^,)]+}})

// LLVM-LABEL: define available_externally void @_ZThn8_N4Impl4bboxE3Strj(
// LLVM-SAME:    ptr dead_on_unwind noalias writable sret(%struct.BBox) align 4 %{{[^,]+}},
// LLVM-SAME:    ptr noundef %{{[^,]+}},
// LLVM-SAME:    ptr nofreeobj noundef align 1 dereferenceable(1) %[[LSSTR:[^,]+]],
// LLVM-SAME:    i32 noundef %{{[^,)]+}})
// LLVM-CIR:     call void @_ZN4Impl4bboxE3Strj(ptr dead_on_unwind writable sret(%struct.BBox) align 4 %{{[^,)]+}}, ptr noundef nonnull align 8 dereferenceable(16) %{{[^,)]+}}, ptr nofreeobj noundef align 1 dereferenceable(1) %[[LSSTR]], i32 noundef %{{[^,)]+}})
// OGCG:         tail call void @_ZN4Impl4bboxE3Strj(ptr dead_on_unwind writable sret(%struct.BBox) align 4 %{{[^,)]+}}, ptr noundef nonnull align 8 dereferenceable(16) %{{[^,)]+}}, ptr nofreeobj noundef align 1 dereferenceable(1) %[[LSSTR]], i32 noundef %{{[^,)]+}})
