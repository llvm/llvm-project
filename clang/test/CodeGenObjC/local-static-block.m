// RUN: %clang_cc1 -fblocks -triple x86_64-apple-darwin -fobjc-runtime=macosx-fragile-10.5 -emit-llvm %s -o %t-64.ll
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.ll %s
// rdar: // 8390455

@class NSArray;

static  NSArray *(^ArrayRecurs)(NSArray *addresses, unsigned long level) = ^(NSArray *addresses, unsigned long level) {

  for(id rawAddress in addresses)
  {
   NSArray *separatedAddresses = ((NSArray*)0);
   separatedAddresses = ArrayRecurs((NSArray *)rawAddress, level+1);
  }
  return (NSArray *)0;
};

void FUNC()
{
 static  NSArray *(^ArrayRecurs)(NSArray *addresses, unsigned long level) = ^(NSArray *addresses, unsigned long level) {

  for(id rawAddress in addresses)
  {
   NSArray *separatedAddresses = ((NSArray*)0);
   separatedAddresses = ArrayRecurs((NSArray *)rawAddress, level+1);
  }
  return (NSArray *)0;
 };

 if (ArrayRecurs) {
   static  NSArray *(^ArrayRecurs)(NSArray *addresses, unsigned long level) = ^(NSArray *addresses, unsigned long level) {

     for(id rawAddress in addresses)
     {
       NSArray *separatedAddresses = ((NSArray*)0);
       separatedAddresses = ArrayRecurs((NSArray *)rawAddress, level+1);
     }
     return (NSArray *)0;
   };
 }
}

void FUNC1()
{
 static  NSArray *(^ArrayRecurs)(NSArray *addresses, unsigned long level) = ^(NSArray *addresses, unsigned long level) {

  for(id rawAddress in addresses)
  {
   NSArray *separatedAddresses = ((NSArray*)0);
   separatedAddresses = ArrayRecurs((NSArray *)rawAddress, level+1);
  }
  return (NSArray *)0;
 };
}
// CHECK-LP64: @ArrayRecurs = internal global
// CHECK-LP64: @FUNC.ArrayRecurs = internal global
// CHECK-LP64: @FUNC.ArrayRecurs3 = internal global
// CHECK-LP64: @FUNC1.ArrayRecurs = internal global
