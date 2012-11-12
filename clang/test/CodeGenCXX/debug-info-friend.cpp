// RUN: %clang -fverbose-asm -S -g %s -o - | grep DW_TAG_friend

class MyFriend;

class SomeClass
{
 friend class MyFriend;
};

SomeClass sc;

