// RUN: %clang_cc1 -triple x86_64-unknown-linux -fsanitize=cfi-icall -fsanitize-trap=cfi-icall -fsanitize-cfi-icall-experimental-normalize-integers -emit-llvm -o - %s | FileCheck %s

// Test that integer types are normalized for cross-language CFI support with
// other languages that can't represent and encode C/C++ integer types.

void foo0(char arg) { }
// CHECK: define{{.*}}foo0{{.*}}!type ![[TYPE0:[0-9]+]] !type !{{[0-9]+}}
void foo1(char arg1, signed char arg2) { }
// CHECK: define{{.*}}foo1{{.*}}!type ![[TYPE1:[0-9]+]] !type !{{[0-9]+}}
void foo2(char arg1, signed char arg2, signed char arg3) { }
// CHECK: define{{.*}}foo2{{.*}}!type ![[TYPE2:[0-9]+]] !type !{{[0-9]+}}
void foo3(int arg) { }
// CHECK: define{{.*}}foo3{{.*}}!type ![[TYPE3:[0-9]+]] !type !{{[0-9]+}}
void foo4(int arg1, int arg2) { }
// CHECK: define{{.*}}foo4{{.*}}!type ![[TYPE4:[0-9]+]] !type !{{[0-9]+}}
void foo5(int arg1, int arg2, int arg3) { }
// CHECK: define{{.*}}foo5{{.*}}!type ![[TYPE5:[0-9]+]] !type !{{[0-9]+}}
void foo6(long arg) { }
// CHECK: define{{.*}}foo6{{.*}}!type ![[TYPE6:[0-9]+]] !type !{{[0-9]+}}
void foo7(long arg1, long long arg2) { }
// CHECK: define{{.*}}foo7{{.*}}!type ![[TYPE7:[0-9]+]] !type !{{[0-9]+}}
void foo8(long arg1, long long arg2, long long arg3) { }
// CHECK: define{{.*}}foo8{{.*}}!type ![[TYPE8:[0-9]+]] !type !{{[0-9]+}}
void foo9(short arg) { }
// CHECK: define{{.*}}foo9{{.*}}!type ![[TYPE9:[0-9]+]] !type !{{[0-9]+}}
void foo10(short arg1, short arg2) { }
// CHECK: define{{.*}}foo10{{.*}}!type ![[TYPE10:[0-9]+]] !type !{{[0-9]+}}
void foo11(short arg1, short arg2, short arg3) { }
// CHECK: define{{.*}}foo11{{.*}}!type ![[TYPE11:[0-9]+]] !type !{{[0-9]+}}
void foo12(unsigned char arg) { }
// CHECK: define{{.*}}foo12{{.*}}!type ![[TYPE12:[0-9]+]] !type !{{[0-9]+}}
void foo13(unsigned char arg1, unsigned char arg2) { }
// CHECK: define{{.*}}foo13{{.*}}!type ![[TYPE13:[0-9]+]] !type !{{[0-9]+}}
void foo14(unsigned char arg1, unsigned char arg2, unsigned char arg3) { }
// CHECK: define{{.*}}foo14{{.*}}!type ![[TYPE14:[0-9]+]] !type !{{[0-9]+}}
void foo15(unsigned int arg) { }
// CHECK: define{{.*}}foo15{{.*}}!type ![[TYPE15:[0-9]+]] !type !{{[0-9]+}}
void foo16(unsigned int arg1, unsigned int arg2) { }
// CHECK: define{{.*}}foo16{{.*}}!type ![[TYPE16:[0-9]+]] !type !{{[0-9]+}}
void foo17(unsigned int arg1, unsigned int arg2, unsigned int arg3) { }
// CHECK: define{{.*}}foo17{{.*}}!type ![[TYPE17:[0-9]+]] !type !{{[0-9]+}}
void foo18(unsigned long arg) { }
// CHECK: define{{.*}}foo18{{.*}}!type ![[TYPE18:[0-9]+]] !type !{{[0-9]+}}
void foo19(unsigned long arg1, unsigned long long arg2) { }
// CHECK: define{{.*}}foo19{{.*}}!type ![[TYPE19:[0-9]+]] !type !{{[0-9]+}}
void foo20(unsigned long arg1, unsigned long long arg2, unsigned long long arg3) { }
// CHECK: define{{.*}}foo20{{.*}}!type ![[TYPE20:[0-9]+]] !type !{{[0-9]+}}
void foo21(unsigned short arg) { }
// CHECK: define{{.*}}foo21{{.*}}!type ![[TYPE21:[0-9]+]] !type !{{[0-9]+}}
void foo22(unsigned short arg1, unsigned short arg2) { }
// CHECK: define{{.*}}foo22{{.*}}!type ![[TYPE22:[0-9]+]] !type !{{[0-9]+}}
void foo23(unsigned short arg1, unsigned short arg2, unsigned short arg3) { }
// CHECK: define{{.*}}foo23{{.*}}!type ![[TYPE23:[0-9]+]] !type !{{[0-9]+}}

// CHECK: ![[TYPE0]] = !{i64 0, !"_ZTSFvu2i8E.normalized"}
// CHECK: ![[TYPE1]] = !{i64 0, !"_ZTSFvu2i8S_E.normalized"}
// CHECK: ![[TYPE2]] = !{i64 0, !"_ZTSFvu2i8S_S_E.normalized"}
// CHECK: ![[TYPE3]] = !{i64 0, !"_ZTSFvu3i32E.normalized"}
// CHECK: ![[TYPE4]] = !{i64 0, !"_ZTSFvu3i32S_E.normalized"}
// CHECK: ![[TYPE5]] = !{i64 0, !"_ZTSFvu3i32S_S_E.normalized"}
// CHECK: ![[TYPE6]] = !{i64 0, !"_ZTSFvu3i64E.normalized"}
// CHECK: ![[TYPE7]] = !{i64 0, !"_ZTSFvu3i64S_E.normalized"}
// CHECK: ![[TYPE8]] = !{i64 0, !"_ZTSFvu3i64S_S_E.normalized"}
// CHECK: ![[TYPE9]] = !{i64 0, !"_ZTSFvu3i16E.normalized"}
// CHECK: ![[TYPE10]] = !{i64 0, !"_ZTSFvu3i16S_E.normalized"}
// CHECK: ![[TYPE11]] = !{i64 0, !"_ZTSFvu3i16S_S_E.normalized"}
// CHECK: ![[TYPE12]] = !{i64 0, !"_ZTSFvu2u8E.normalized"}
// CHECK: ![[TYPE13]] = !{i64 0, !"_ZTSFvu2u8S_E.normalized"}
// CHECK: ![[TYPE14]] = !{i64 0, !"_ZTSFvu2u8S_S_E.normalized"}
// CHECK: ![[TYPE15]] = !{i64 0, !"_ZTSFvu3u32E.normalized"}
// CHECK: ![[TYPE16]] = !{i64 0, !"_ZTSFvu3u32S_E.normalized"}
// CHECK: ![[TYPE17]] = !{i64 0, !"_ZTSFvu3u32S_S_E.normalized"}
// CHECK: ![[TYPE18]] = !{i64 0, !"_ZTSFvu3u64E.normalized"}
// CHECK: ![[TYPE19]] = !{i64 0, !"_ZTSFvu3u64S_E.normalized"}
// CHECK: ![[TYPE20]] = !{i64 0, !"_ZTSFvu3u64S_S_E.normalized"}
// CHECK: ![[TYPE21]] = !{i64 0, !"_ZTSFvu3u16E.normalized"}
// CHECK: ![[TYPE22]] = !{i64 0, !"_ZTSFvu3u16S_E.normalized"}
// CHECK: ![[TYPE23]] = !{i64 0, !"_ZTSFvu3u16S_S_E.normalized"}
