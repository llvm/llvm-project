// RUN: %clang_cc1 -S -o - -triple i686-windows -verify -fblocks \
// RUN:     -Wno-unused-comparison %s

#pragma clang diagnostic ignored "-Wunused-comparison"

#define nil ((id)nullptr)

@protocol NSObject
@end

@protocol NSCopying
@end

@protocol OtherProtocol
@end

__attribute__((objc_root_class))
@interface NSObject <NSObject, NSCopying>
@end

__attribute__((objc_root_class))
@interface Test
@end

int main() {
  void (^block)() = ^{};
  NSObject *object;
  id<NSObject, NSCopying> qualifiedId;

  id<OtherProtocol> poorlyQualified1;
  Test *objectOfWrongType;

  block == nil;
  block == object;
  block == qualifiedId;

  nil == block;
  object == block;
  qualifiedId == block;

  // these are still not valid: blocks must be compared with id, NSObject*, or a protocol-qualified id
  // conforming to NSCopying or NSObject.

  block == poorlyQualified1; // expected-error {{invalid operands to binary expression ('void (^)()' and 'id<OtherProtocol>')}}
  block == objectOfWrongType; // expected-error {{invalid operands to binary expression ('void (^)()' and 'Test *')}}

  poorlyQualified1 == block; // expected-error {{invalid operands to binary expression ('id<OtherProtocol>' and 'void (^)()')}}
  objectOfWrongType == block; // expected-error {{invalid operands to binary expression ('Test *' and 'void (^)()')}}

  return 0;
}
