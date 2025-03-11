// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -debug-info-kind=standalone -O0 \
// RUN:     -emit-llvm  -fexperimental-assignment-tracking=forced %s -o -        \
// RUN:     -disable-O0-optnone                                                  \
// RUN: | FileCheck %s

// Check that the (debug) codegen looks right with assignment tracking
// enabled. Each fragment that is written to should have a dbg.assign that has
// the DIAssignID of the write as an argument. The fragment offset and size
// should match the offset into the base storage and size of the store. Each of
// the scenarios below results in slightly different arguments generated for
// the memcpy.

// Test write a complete struct field only.
void fragmentWhole()
{
 struct Record {
   int num;
   char ch;
 };

 Record dest;
 char src = '\0';
 __builtin_memcpy(&dest.ch, &src, sizeof(char));
}
// CHECK: call void @llvm.memcpy{{.+}}, !DIAssignID ![[memberID:[0-9]+]]
// CHECK-NEXT: #dbg_assign({{.*}}poison, !{{[0-9]+}}, !DIExpression(DW_OP_LLVM_fragment, 32, 8), ![[memberID]], ptr %ch, !DIExpression(),

// Write starting at a field and overlapping part of another.
void fragmentWholeToPartial()
{
 struct Record {
   int num1;
   int num2;
 };

 Record dest;
 char src[5]="\0\0\0\0";
 __builtin_memcpy(&dest.num1, &src, 5);
}
// CHECK: call void @llvm.memcpy{{.+}}, !DIAssignID ![[exceed:[0-9]+]]
// CHECK-NEXT: #dbg_assign({{.*}}poison, !{{[0-9]+}}, !DIExpression(DW_OP_LLVM_fragment, 0, 40), ![[exceed]], ptr %num1, !DIExpression(),

// Write starting between fields.
void fragmentPartialToWhole()
{
 struct record {
   int num1;
   int num2;
   int num3;
};

 record dest;
 char src[5]="\0\0\0\0";
 __builtin_memcpy((char*)&(dest.num2) + 3, &src, 5);
}
// CHECK: call void @llvm.memcpy{{.+}}, !DIAssignID ![[addendID:[0-9]+]]
// CHECK-NEXT: #dbg_assign({{.*}}poison, !{{.*}}, !DIExpression(DW_OP_LLVM_fragment, 56, 40), ![[addendID]], ptr %add.ptr, !DIExpression(),
