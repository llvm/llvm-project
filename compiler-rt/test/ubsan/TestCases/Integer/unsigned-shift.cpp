// RUN: %clangxx -fsanitize=unsigned-shift-base %s -o %t1 && not %run %t1 2>&1 | FileCheck %s
// RUN: %clangxx -fsanitize=unsigned-shift-base,shift-exponent %s -o %t1 && not %run %t1 2>&1 | FileCheck %s

#define shift(val, amount) ({      \
  volatile unsigned _v = (val);    \
  volatile unsigned _a = (amount); \
  unsigned res = _v << _a;         \
  res;                             \
})

int main() {

  shift(0b00000000'00000000'00000000'00000000, 31);
  shift(0b00000000'00000000'00000000'00000001, 31);
  shift(0b00000000'00000000'00000000'00000010, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 2 by 31 places cannot be represented in type 'unsigned int'
  shift(0b00000000'00000000'00000000'00000100, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 4 by 31 places cannot be represented in type 'unsigned int'
  shift(0b00000000'00000000'00000000'00001000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 8 by 31 places cannot be represented in type 'unsigned int'
  shift(0b00000000'00000000'00000000'00010000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 16 by 31 places cannot be represented in type 'unsigned int'
  shift(0b00000000'00000000'00000000'00100000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 32 by 31 places cannot be represented in type 'unsigned int'
  shift(0b00000000'00000000'00000000'01000000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 64 by 31 places cannot be represented in type 'unsigned int'
  shift(0b00000000'00000000'00000000'10000000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 128 by 31 places cannot be represented in type 'unsigned int'
  shift(0b00000000'00000000'00000001'00000000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 256 by 31 places cannot be represented in type 'unsigned int'
  shift(0b00000000'00000000'00000010'00000000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 512 by 31 places cannot be represented in type 'unsigned int'
  shift(0b00000000'00000000'00000100'00000000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 1024 by 31 places cannot be represented in type 'unsigned int'
  shift(0b00000000'00000000'00001000'00000000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 2048 by 31 places cannot be represented in type 'unsigned int'
  shift(0b00000000'00000000'00010000'00000000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 4096 by 31 places cannot be represented in type 'unsigned int'
  shift(0b00000000'00000000'00100000'00000000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 8192 by 31 places cannot be represented in type 'unsigned int'
  shift(0b00000000'00000000'01000000'00000000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 16384 by 31 places cannot be represented in type 'unsigned int'
  shift(0b00000000'00000000'10000000'00000000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 32768 by 31 places cannot be represented in type 'unsigned int'
  shift(0b00000000'00000001'00000000'00000000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 65536 by 31 places cannot be represented in type 'unsigned int'
  shift(0b00000000'00000010'00000000'00000000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 131072 by 31 places cannot be represented in type 'unsigned int'
  shift(0b00000000'00000100'00000000'00000000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 262144 by 31 places cannot be represented in type 'unsigned int'
  shift(0b00000000'00001000'00000000'00000000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 524288 by 31 places cannot be represented in type 'unsigned int'
  shift(0b00000000'00010000'00000000'00000000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 1048576 by 31 places cannot be represented in type 'unsigned int'
  shift(0b00000000'00100000'00000000'00000000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 2097152 by 31 places cannot be represented in type 'unsigned int'
  shift(0b00000000'01000000'00000000'00000000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 4194304 by 31 places cannot be represented in type 'unsigned int'
  shift(0b00000000'10000000'00000000'00000000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 8388608 by 31 places cannot be represented in type 'unsigned int'
  shift(0b00000001'00000000'00000000'00000000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 16777216 by 31 places cannot be represented in type 'unsigned int'
  shift(0b00000010'00000000'00000000'00000000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 33554432 by 31 places cannot be represented in type 'unsigned int'
  shift(0b00000100'00000000'00000000'00000000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 67108864 by 31 places cannot be represented in type 'unsigned int'
  shift(0b00001000'00000000'00000000'00000000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 134217728 by 31 places cannot be represented in type 'unsigned int'
  shift(0b00010000'00000000'00000000'00000000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 268435456 by 31 places cannot be represented in type 'unsigned int'
  shift(0b00100000'00000000'00000000'00000000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 536870912 by 31 places cannot be represented in type 'unsigned int'
  shift(0b01000000'00000000'00000000'00000000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 1073741824 by 31 places cannot be represented in type 'unsigned int'
  shift(0b10000000'00000000'00000000'00000000, 31); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 2147483648 by 31 places cannot be represented in type 'unsigned int'

  shift(0b10000000'00000000'00000000'00000000, 00);
  shift(0b10000000'00000000'00000000'00000000, 01); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 2147483648 by 1 places cannot be represented in type 'unsigned int'

  shift(0xffff'ffff, 0);
  shift(0xffff'ffff, 1); // CHECK: unsigned-shift.cpp:[[@LINE]]:3: runtime error: left shift of 4294967295 by 1 places cannot be represented in type 'unsigned int'

  return 1;
}
