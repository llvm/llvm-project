// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -fblocks -Wno-address-of-temporary -D"Class=void*" -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// rdar://11202764

typedef void(^BL)(void);

id return_id(void(^block)(void)) {
  return block;
}

BL return_block(id obj) {
  return obj;
}

int main()
{
    void(^block)(void);
    id obj;
    block = obj; // AnyPointerToBlockPointerCast
    obj = block; // BlockPointerToObjCPointerCast

   id obj1 = block;

   void(^block1)(void) = obj1;

   return_id(block1);

   return_id(obj1);

   return_block(block1);

   return_block(obj1);
}
