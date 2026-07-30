// RUN: %clang_cc1 -fblocks -triple x86_64-apple-darwin -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - %s | FileCheck %s

// CHECK: @ArrayRecurs = internal global
// CHECK: @FUNC.ArrayRecurs = internal global
// CHECK: @FUNC.ArrayRecurs.1 = internal global
// CHECK: @FUNC1.ArrayRecurs = internal global

@class NSArray;

static  NSArray *(^ArrayRecurs)(NSArray *addresses, unsigned long level) = ^(NSArray *addresses, unsigned long level) {

  for(id rawAddress in addresses)
  {
   NSArray *separatedAddresses = ((NSArray*)0);
   separatedAddresses = ArrayRecurs((NSArray *)rawAddress, level+1);
  }
  return (NSArray *)0;
};

extern NSArray *address;
extern unsigned long level;

void FUNC(void)
{
 ArrayRecurs(address, level);

 static  NSArray *(^ArrayRecurs)(NSArray *addresses, unsigned long level) = ^(NSArray *addresses, unsigned long level) {

  for(id rawAddress in addresses)
  {
   NSArray *separatedAddresses = ((NSArray*)0);
   separatedAddresses = ArrayRecurs((NSArray *)rawAddress, level+1);
  }
  return (NSArray *)0;
 };
 ArrayRecurs(address, level);

 if (ArrayRecurs) {
   static  NSArray *(^ArrayRecurs)(NSArray *addresses, unsigned long level) = ^(NSArray *addresses, unsigned long level) {

     for(id rawAddress in addresses)
     {
       NSArray *separatedAddresses = ((NSArray*)0);
       separatedAddresses = ArrayRecurs((NSArray *)rawAddress, level+1);
     }
     return (NSArray *)0;
   };
   ArrayRecurs(address, level);
 }
}

void FUNC2(void) {
  static void (^const block1)(int) = ^(int a){
    if (a--)
      block1(a);
  };
}

// CHECK-LABEL: define{{.*}} void @FUNC2(
// CHECK: define internal void @_block_invoke{{.*}}(
// CHECK: call void %{{.*}}(ptr noundef @__block_literal_global{{.*}}, i32 noundef %{{.*}})

void FUNC1(void)
{
 static  NSArray *(^ArrayRecurs)(NSArray *addresses, unsigned long level) = ^(NSArray *addresses, unsigned long level) {

  for(id rawAddress in addresses)
  {
   NSArray *separatedAddresses = ((NSArray*)0);
   separatedAddresses = ArrayRecurs((NSArray *)rawAddress, level+1);
  }
  return (NSArray *)0;
 };
 ArrayRecurs(address, level);
}
