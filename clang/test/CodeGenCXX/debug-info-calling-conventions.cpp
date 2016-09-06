// RUN: %clang_cc1 %s -triple=i686-pc-windows-msvc -debug-info-kind=limited -emit-llvm -o - | FileCheck %s

struct A {
  void thiscallcc();
};
void A::thiscallcc() {}

// CHECK: !DISubprogram(name: "thiscallcc", {{.*}} type: ![[thiscallty:[^,]*]], {{.*}})
// CHECK: ![[thiscallty]] = !DISubroutineType(cc: DW_CC_BORLAND_thiscall, types: ![[thisargs:[^,)]*]])
// CHECK: ![[thisargs]] = !{null, ![[thisptrty:[^,}]*]]}
// CHECK: ![[thisptrty]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !{{.*}}, size: 32, align: 32, flags: DIFlagArtificial | DIFlagObjectPointer)

void cdeclcc() {}
void __fastcall fastcallcc() {}
void __stdcall stdcallcc() {}
void __vectorcall vectorcallcc() {}

// CHECK: !DISubprogram(name: "cdeclcc", {{.*}} type: ![[cdeclty:[^,]*]], {{.*}})
// CHECK: ![[cdeclty]] = !DISubroutineType(types: ![[noargs:[^,)]*]])
// CHECK: ![[noargs]] = !{null}
// CHECK: !DISubprogram(name: "fastcallcc", {{.*}} type: ![[fastcallty:[^,]*]], {{.*}})
// CHECK: ![[fastcallty]] = !DISubroutineType(cc: DW_CC_BORLAND_msfastcall, types: ![[noargs]])
// CHECK: !DISubprogram(name: "stdcallcc", {{.*}} type: ![[stdcallty:[^,]*]], {{.*}})
// CHECK: ![[stdcallty]] = !DISubroutineType(cc: DW_CC_BORLAND_stdcall, types: ![[noargs]])
// CHECK: !DISubprogram(name: "vectorcallcc", {{.*}} type: ![[vectorcallty:[^,]*]], {{.*}})
// CHECK: ![[vectorcallty]] = !DISubroutineType(cc: DW_CC_LLVM_vectorcall, types: ![[noargs]])
