// RUN: %clang_analyze_cc1 -triple x86_64-pc-linux-gnu -analyzer-checker=debug.ExprInspection -analyzer-config support-symbolic-integer-casts=true -verify %s

using uchar = unsigned char;
using schar = signed char;
using ushort = unsigned short;
using uint = unsigned int;
using ulong = unsigned long;
using llong = long long;
using ullong = unsigned long long;

template <typename T>
void clang_analyzer_dump(T);

void test_schar(schar x) {
  clang_analyzer_dump(x); // expected-warning{{reg_$0<schar x>}}

  clang_analyzer_dump((schar)x);  // expected-warning{{reg_$0<schar x>}}
  clang_analyzer_dump((char)x);   // expected-warning{{(char) (reg_$0<schar x>)}}
  clang_analyzer_dump((short)x);  // expected-warning{{(short) (reg_$0<schar x>)}}
  clang_analyzer_dump((int)x);    // expected-warning{{(int) (reg_$0<schar x>)}}
  clang_analyzer_dump((long)x);   // expected-warning{{(long) (reg_$0<schar x>)}}
  clang_analyzer_dump((llong)x);  // expected-warning{{(long long) (reg_$0<schar x>)}}
  clang_analyzer_dump((uchar)x);  // expected-warning{{(unsigned char) (reg_$0<schar x>)}}
  clang_analyzer_dump((ushort)x); // expected-warning{{(unsigned short) (reg_$0<schar x>)}}
  clang_analyzer_dump((uint)x);   // expected-warning{{(unsigned int) (reg_$0<schar x>)}}
  clang_analyzer_dump((ulong)x);  // expected-warning{{(unsigned long) (reg_$0<schar x>)}}
  clang_analyzer_dump((ullong)x); // expected-warning{{(unsigned long long) (reg_$0<schar x>)}}

  clang_analyzer_dump((schar)(schar)x);  // expected-warning{{reg_$0<schar x>}}
  clang_analyzer_dump((schar)(char)x);   // expected-warning{{reg_$0<schar x>}}
  clang_analyzer_dump((schar)(short)x);  // expected-warning{{reg_$0<schar x>}}
  clang_analyzer_dump((schar)(int)x);    // expected-warning{{reg_$0<schar x>}}
  clang_analyzer_dump((schar)(long)x);   // expected-warning{{reg_$0<schar x>}}
  clang_analyzer_dump((schar)(llong)x);  // expected-warning{{reg_$0<schar x>}}
  clang_analyzer_dump((schar)(uchar)x);  // expected-warning{{reg_$0<schar x>}}
  clang_analyzer_dump((schar)(ushort)x); // expected-warning{{reg_$0<schar x>}}
  clang_analyzer_dump((schar)(uint)x);   // expected-warning{{reg_$0<schar x>}}
  clang_analyzer_dump((schar)(ulong)x);  // expected-warning{{reg_$0<schar x>}}
  clang_analyzer_dump((schar)(ullong)x); // expected-warning{{reg_$0<schar x>}}

  clang_analyzer_dump((char)(schar)x);  // expected-warning{{(char) (reg_$0<schar x>)}}
  clang_analyzer_dump((char)(char)x);   // expected-warning{{(char) (reg_$0<schar x>)}}
  clang_analyzer_dump((char)(short)x);  // expected-warning{{(char) (reg_$0<schar x>)}}
  clang_analyzer_dump((char)(int)x);    // expected-warning{{(char) (reg_$0<schar x>)}}
  clang_analyzer_dump((char)(long)x);   // expected-warning{{(char) (reg_$0<schar x>)}}
  clang_analyzer_dump((char)(llong)x);  // expected-warning{{(char) (reg_$0<schar x>)}}
  clang_analyzer_dump((char)(uchar)x);  // expected-warning{{(char) (reg_$0<schar x>)}}
  clang_analyzer_dump((char)(ushort)x); // expected-warning{{(char) (reg_$0<schar x>)}}
  clang_analyzer_dump((char)(uint)x);   // expected-warning{{(char) (reg_$0<schar x>)}}
  clang_analyzer_dump((char)(ulong)x);  // expected-warning{{(char) (reg_$0<schar x>)}}
  clang_analyzer_dump((char)(ullong)x); // expected-warning{{(char) (reg_$0<schar x>)}}

  clang_analyzer_dump((short)(schar)x);  // expected-warning{{(short) (reg_$0<schar x>)}}
  clang_analyzer_dump((short)(char)x);   // expected-warning{{(short) (reg_$0<schar x>)}}
  clang_analyzer_dump((short)(short)x);  // expected-warning{{(short) (reg_$0<schar x>)}}
  clang_analyzer_dump((short)(int)x);    // expected-warning{{(short) (reg_$0<schar x>)}}
  clang_analyzer_dump((short)(long)x);   // expected-warning{{(short) (reg_$0<schar x>)}}
  clang_analyzer_dump((short)(llong)x);  // expected-warning{{(short) (reg_$0<schar x>)}}
  clang_analyzer_dump((short)(uchar)x);  // expected-warning{{(short) ((unsigned char) (reg_$0<schar x>))}}
  clang_analyzer_dump((short)(ushort)x); // expected-warning{{(short) (reg_$0<schar x>)}}
  clang_analyzer_dump((short)(uint)x);   // expected-warning{{(short) (reg_$0<schar x>)}}
  clang_analyzer_dump((short)(ulong)x);  // expected-warning{{(short) (reg_$0<schar x>)}}
  clang_analyzer_dump((short)(ullong)x); // expected-warning{{(short) (reg_$0<schar x>)}}

  clang_analyzer_dump((int)(schar)x);  // expected-warning{{(int) (reg_$0<schar x>)}}
  clang_analyzer_dump((int)(char)x);   // expected-warning{{(int) (reg_$0<schar x>)}}
  clang_analyzer_dump((int)(short)x);  // expected-warning{{(int) (reg_$0<schar x>)}}
  clang_analyzer_dump((int)(int)x);    // expected-warning{{(int) (reg_$0<schar x>)}}
  clang_analyzer_dump((int)(long)x);   // expected-warning{{(int) (reg_$0<schar x>)}}
  clang_analyzer_dump((int)(llong)x);  // expected-warning{{(int) (reg_$0<schar x>)}}
  clang_analyzer_dump((int)(uchar)x);  // expected-warning{{(int) ((unsigned char) (reg_$0<schar x>))}}
  clang_analyzer_dump((int)(ushort)x); // expected-warning{{(int) ((unsigned short) (reg_$0<schar x>))}}
  clang_analyzer_dump((int)(uint)x);   // expected-warning{{(int) (reg_$0<schar x>)}}
  clang_analyzer_dump((int)(ulong)x);  // expected-warning{{(int) (reg_$0<schar x>)}}
  clang_analyzer_dump((int)(ullong)x); // expected-warning{{(int) (reg_$0<schar x>)}}

  clang_analyzer_dump((long)(schar)x);  // expected-warning{{(long) (reg_$0<schar x>)}}
  clang_analyzer_dump((long)(char)x);   // expected-warning{{(long) (reg_$0<schar x>)}}
  clang_analyzer_dump((long)(short)x);  // expected-warning{{(long) (reg_$0<schar x>)}}
  clang_analyzer_dump((long)(int)x);    // expected-warning{{(long) (reg_$0<schar x>)}}
  clang_analyzer_dump((long)(long)x);   // expected-warning{{(long) (reg_$0<schar x>)}}
  clang_analyzer_dump((long)(llong)x);  // expected-warning{{(long) (reg_$0<schar x>)}}
  clang_analyzer_dump((long)(uchar)x);  // expected-warning{{(long) ((unsigned char) (reg_$0<schar x>))}}
  clang_analyzer_dump((long)(ushort)x); // expected-warning{{(long) ((unsigned short) (reg_$0<schar x>))}}
  clang_analyzer_dump((long)(uint)x);   // expected-warning{{(long) ((unsigned int) (reg_$0<schar x>))}}
  clang_analyzer_dump((long)(ulong)x);  // expected-warning{{(long) (reg_$0<schar x>)}}
  clang_analyzer_dump((long)(ullong)x); // expected-warning{{(long) (reg_$0<schar x>)}}

  clang_analyzer_dump((llong)(schar)x);  // expected-warning{{(long long) (reg_$0<schar x>)}}
  clang_analyzer_dump((llong)(char)x);   // expected-warning{{(long long) (reg_$0<schar x>)}}
  clang_analyzer_dump((llong)(short)x);  // expected-warning{{(long long) (reg_$0<schar x>)}}
  clang_analyzer_dump((llong)(int)x);    // expected-warning{{(long long) (reg_$0<schar x>)}}
  clang_analyzer_dump((llong)(long)x);   // expected-warning{{(long long) (reg_$0<schar x>)}}
  clang_analyzer_dump((llong)(llong)x);  // expected-warning{{(long long) (reg_$0<schar x>)}}
  clang_analyzer_dump((llong)(uchar)x);  // expected-warning{{(long long) ((unsigned char) (reg_$0<schar x>))}}
  clang_analyzer_dump((llong)(ushort)x); // expected-warning{{(long long) ((unsigned short) (reg_$0<schar x>))}}
  clang_analyzer_dump((llong)(uint)x);   // expected-warning{{(long long) ((unsigned int) (reg_$0<schar x>))}}
  clang_analyzer_dump((llong)(ulong)x);  // expected-warning{{(long long) (reg_$0<schar x>)}}
  clang_analyzer_dump((llong)(ullong)x); // expected-warning{{(long long) (reg_$0<schar x>)}}

  clang_analyzer_dump((uchar)(schar)x);  // expected-warning{{(unsigned char) (reg_$0<schar x>)}}
  clang_analyzer_dump((uchar)(char)x);   // expected-warning{{(unsigned char) (reg_$0<schar x>)}}
  clang_analyzer_dump((uchar)(short)x);  // expected-warning{{(unsigned char) (reg_$0<schar x>)}}
  clang_analyzer_dump((uchar)(int)x);    // expected-warning{{(unsigned char) (reg_$0<schar x>)}}
  clang_analyzer_dump((uchar)(long)x);   // expected-warning{{(unsigned char) (reg_$0<schar x>)}}
  clang_analyzer_dump((uchar)(llong)x);  // expected-warning{{(unsigned char) (reg_$0<schar x>)}}
  clang_analyzer_dump((uchar)(uchar)x);  // expected-warning{{(unsigned char) (reg_$0<schar x>)}}
  clang_analyzer_dump((uchar)(ushort)x); // expected-warning{{(unsigned char) (reg_$0<schar x>)}}
  clang_analyzer_dump((uchar)(uint)x);   // expected-warning{{(unsigned char) (reg_$0<schar x>)}}
  clang_analyzer_dump((uchar)(ulong)x);  // expected-warning{{(unsigned char) (reg_$0<schar x>)}}
  clang_analyzer_dump((uchar)(ullong)x); // expected-warning{{(unsigned char) (reg_$0<schar x>)}}

  clang_analyzer_dump((ushort)(schar)x);  // expected-warning{{(unsigned short) (reg_$0<schar x>)}}
  clang_analyzer_dump((ushort)(char)x);   // expected-warning{{(unsigned short) (reg_$0<schar x>)}}
  clang_analyzer_dump((ushort)(short)x);  // expected-warning{{(unsigned short) (reg_$0<schar x>)}}
  clang_analyzer_dump((ushort)(int)x);    // expected-warning{{(unsigned short) (reg_$0<schar x>)}}
  clang_analyzer_dump((ushort)(long)x);   // expected-warning{{(unsigned short) (reg_$0<schar x>)}}
  clang_analyzer_dump((ushort)(llong)x);  // expected-warning{{(unsigned short) (reg_$0<schar x>)}}
  clang_analyzer_dump((ushort)(uchar)x);  // expected-warning{{(unsigned short) ((unsigned char) (reg_$0<schar x>))}}
  clang_analyzer_dump((ushort)(ushort)x); // expected-warning{{(unsigned short) (reg_$0<schar x>)}}
  clang_analyzer_dump((ushort)(uint)x);   // expected-warning{{(unsigned short) (reg_$0<schar x>)}}
  clang_analyzer_dump((ushort)(ulong)x);  // expected-warning{{(unsigned short) (reg_$0<schar x>)}}
  clang_analyzer_dump((ushort)(ullong)x); // expected-warning{{(unsigned short) (reg_$0<schar x>)}}

  clang_analyzer_dump((uint)(schar)x);  // expected-warning{{(unsigned int) (reg_$0<schar x>)}}
  clang_analyzer_dump((uint)(char)x);   // expected-warning{{(unsigned int) (reg_$0<schar x>)}}
  clang_analyzer_dump((uint)(short)x);  // expected-warning{{(unsigned int) (reg_$0<schar x>)}}
  clang_analyzer_dump((uint)(int)x);    // expected-warning{{(unsigned int) (reg_$0<schar x>)}}
  clang_analyzer_dump((uint)(long)x);   // expected-warning{{(unsigned int) (reg_$0<schar x>)}}
  clang_analyzer_dump((uint)(llong)x);  // expected-warning{{(unsigned int) (reg_$0<schar x>)}}
  clang_analyzer_dump((uint)(uchar)x);  // expected-warning{{(unsigned int) ((unsigned char) (reg_$0<schar x>))}}
  clang_analyzer_dump((uint)(ushort)x); // expected-warning{{(unsigned int) ((unsigned short) (reg_$0<schar x>))}}
  clang_analyzer_dump((uint)(uint)x);   // expected-warning{{(unsigned int) (reg_$0<schar x>)}}
  clang_analyzer_dump((uint)(ulong)x);  // expected-warning{{(unsigned int) (reg_$0<schar x>)}}
  clang_analyzer_dump((uint)(ullong)x); // expected-warning{{(unsigned int) (reg_$0<schar x>)}}

  clang_analyzer_dump((ulong)(schar)x);  // expected-warning{{(unsigned long) (reg_$0<schar x>)}}
  clang_analyzer_dump((ulong)(char)x);   // expected-warning{{(unsigned long) (reg_$0<schar x>)}}
  clang_analyzer_dump((ulong)(short)x);  // expected-warning{{(unsigned long) (reg_$0<schar x>)}}
  clang_analyzer_dump((ulong)(int)x);    // expected-warning{{(unsigned long) (reg_$0<schar x>)}}
  clang_analyzer_dump((ulong)(long)x);   // expected-warning{{(unsigned long) (reg_$0<schar x>)}}
  clang_analyzer_dump((ulong)(llong)x);  // expected-warning{{(unsigned long) (reg_$0<schar x>)}}
  clang_analyzer_dump((ulong)(uchar)x);  // expected-warning{{(unsigned long) ((unsigned char) (reg_$0<schar x>))}}
  clang_analyzer_dump((ulong)(ushort)x); // expected-warning{{(unsigned long) ((unsigned short) (reg_$0<schar x>))}}
  clang_analyzer_dump((ulong)(uint)x);   // expected-warning{{(unsigned long) ((unsigned int) (reg_$0<schar x>))}}
  clang_analyzer_dump((ulong)(ulong)x);  // expected-warning{{(unsigned long) (reg_$0<schar x>)}}
  clang_analyzer_dump((ulong)(ullong)x); // expected-warning{{(unsigned long) (reg_$0<schar x>)}}

  clang_analyzer_dump((ullong)(schar)x);  // expected-warning{{(unsigned long long) (reg_$0<schar x>)}}
  clang_analyzer_dump((ullong)(char)x);   // expected-warning{{(unsigned long long) (reg_$0<schar x>)}}
  clang_analyzer_dump((ullong)(short)x);  // expected-warning{{(unsigned long long) (reg_$0<schar x>)}}
  clang_analyzer_dump((ullong)(int)x);    // expected-warning{{(unsigned long long) (reg_$0<schar x>)}}
  clang_analyzer_dump((ullong)(long)x);   // expected-warning{{(unsigned long long) (reg_$0<schar x>)}}
  clang_analyzer_dump((ullong)(llong)x);  // expected-warning{{(unsigned long long) (reg_$0<schar x>)}}
  clang_analyzer_dump((ullong)(uchar)x);  // expected-warning{{(unsigned long long) ((unsigned char) (reg_$0<schar x>))}}
  clang_analyzer_dump((ullong)(ushort)x); // expected-warning{{(unsigned long long) ((unsigned short) (reg_$0<schar x>))}}
  clang_analyzer_dump((ullong)(uint)x);   // expected-warning{{(unsigned long long) ((unsigned int) (reg_$0<schar x>))}}
  clang_analyzer_dump((ullong)(ulong)x);  // expected-warning{{(unsigned long long) (reg_$0<schar x>)}}
  clang_analyzer_dump((ullong)(ullong)x); // expected-warning{{(unsigned long long) (reg_$0<schar x>)}}
}

void test_char(char x) {
  clang_analyzer_dump(x); // expected-warning{{reg_$0<char x>}}

  clang_analyzer_dump((schar)x);  // expected-warning{{(signed char) (reg_$0<char x>)}}
  clang_analyzer_dump((char)x);   // expected-warning{{reg_$0<char x>}}
  clang_analyzer_dump((short)x);  // expected-warning{{(short) (reg_$0<char x>)}}
  clang_analyzer_dump((int)x);    // expected-warning{{(int) (reg_$0<char x>)}}
  clang_analyzer_dump((long)x);   // expected-warning{{(long) (reg_$0<char x>)}}
  clang_analyzer_dump((llong)x);  // expected-warning{{(long long) (reg_$0<char x>)}}
  clang_analyzer_dump((uchar)x);  // expected-warning{{(unsigned char) (reg_$0<char x>)}}
  clang_analyzer_dump((ushort)x); // expected-warning{{(unsigned short) (reg_$0<char x>)}}
  clang_analyzer_dump((uint)x);   // expected-warning{{(unsigned int) (reg_$0<char x>)}}
  clang_analyzer_dump((ulong)x);  // expected-warning{{(unsigned long) (reg_$0<char x>)}}
  clang_analyzer_dump((ullong)x); // expected-warning{{(unsigned long long) (reg_$0<char x>)}}

  clang_analyzer_dump((schar)(schar)x);  // expected-warning{{(signed char) (reg_$0<char x>)}}
  clang_analyzer_dump((schar)(char)x);   // expected-warning{{(signed char) (reg_$0<char x>)}}
  clang_analyzer_dump((schar)(short)x);  // expected-warning{{(signed char) (reg_$0<char x>)}}
  clang_analyzer_dump((schar)(int)x);    // expected-warning{{(signed char) (reg_$0<char x>)}}
  clang_analyzer_dump((schar)(long)x);   // expected-warning{{(signed char) (reg_$0<char x>)}}
  clang_analyzer_dump((schar)(llong)x);  // expected-warning{{(signed char) (reg_$0<char x>)}}
  clang_analyzer_dump((schar)(uchar)x);  // expected-warning{{(signed char) (reg_$0<char x>)}}
  clang_analyzer_dump((schar)(ushort)x); // expected-warning{{(signed char) (reg_$0<char x>)}}
  clang_analyzer_dump((schar)(uint)x);   // expected-warning{{(signed char) (reg_$0<char x>)}}
  clang_analyzer_dump((schar)(ulong)x);  // expected-warning{{(signed char) (reg_$0<char x>)}}
  clang_analyzer_dump((schar)(ullong)x); // expected-warning{{(signed char) (reg_$0<char x>)}}

  clang_analyzer_dump((char)(schar)x);  // expected-warning{{reg_$0<char x>}}
  clang_analyzer_dump((char)(char)x);   // expected-warning{{reg_$0<char x>}}
  clang_analyzer_dump((char)(short)x);  // expected-warning{{reg_$0<char x>}}
  clang_analyzer_dump((char)(int)x);    // expected-warning{{reg_$0<char x>}}
  clang_analyzer_dump((char)(long)x);   // expected-warning{{reg_$0<char x>}}
  clang_analyzer_dump((char)(llong)x);  // expected-warning{{reg_$0<char x>}}
  clang_analyzer_dump((char)(uchar)x);  // expected-warning{{reg_$0<char x>}}
  clang_analyzer_dump((char)(ushort)x); // expected-warning{{reg_$0<char x>}}
  clang_analyzer_dump((char)(uint)x);   // expected-warning{{reg_$0<char x>}}
  clang_analyzer_dump((char)(ulong)x);  // expected-warning{{reg_$0<char x>}}
  clang_analyzer_dump((char)(ullong)x); // expected-warning{{reg_$0<char x>}}

  clang_analyzer_dump((short)(schar)x);  // expected-warning{{(short) (reg_$0<char x>)}}
  clang_analyzer_dump((short)(char)x);   // expected-warning{{(short) (reg_$0<char x>)}}
  clang_analyzer_dump((short)(short)x);  // expected-warning{{(short) (reg_$0<char x>)}}
  clang_analyzer_dump((short)(int)x);    // expected-warning{{(short) (reg_$0<char x>)}}
  clang_analyzer_dump((short)(long)x);   // expected-warning{{(short) (reg_$0<char x>)}}
  clang_analyzer_dump((short)(llong)x);  // expected-warning{{(short) (reg_$0<char x>)}}
  clang_analyzer_dump((short)(uchar)x);  // expected-warning{{(short) ((unsigned char) (reg_$0<char x>))}}
  clang_analyzer_dump((short)(ushort)x); // expected-warning{{(short) (reg_$0<char x>)}}
  clang_analyzer_dump((short)(uint)x);   // expected-warning{{(short) (reg_$0<char x>)}}
  clang_analyzer_dump((short)(ulong)x);  // expected-warning{{(short) (reg_$0<char x>)}}
  clang_analyzer_dump((short)(ullong)x); // expected-warning{{(short) (reg_$0<char x>)}}

  clang_analyzer_dump((int)(schar)x);  // expected-warning{{(int) (reg_$0<char x>)}}
  clang_analyzer_dump((int)(char)x);   // expected-warning{{(int) (reg_$0<char x>)}}
  clang_analyzer_dump((int)(short)x);  // expected-warning{{(int) (reg_$0<char x>)}}
  clang_analyzer_dump((int)(int)x);    // expected-warning{{(int) (reg_$0<char x>)}}
  clang_analyzer_dump((int)(long)x);   // expected-warning{{(int) (reg_$0<char x>)}}
  clang_analyzer_dump((int)(llong)x);  // expected-warning{{(int) (reg_$0<char x>)}}
  clang_analyzer_dump((int)(uchar)x);  // expected-warning{{(int) ((unsigned char) (reg_$0<char x>))}}
  clang_analyzer_dump((int)(ushort)x); // expected-warning{{(int) ((unsigned short) (reg_$0<char x>))}}
  clang_analyzer_dump((int)(uint)x);   // expected-warning{{(int) (reg_$0<char x>)}}
  clang_analyzer_dump((int)(ulong)x);  // expected-warning{{(int) (reg_$0<char x>)}}
  clang_analyzer_dump((int)(ullong)x); // expected-warning{{(int) (reg_$0<char x>)}}

  clang_analyzer_dump((long)(schar)x);  // expected-warning{{(long) (reg_$0<char x>)}}
  clang_analyzer_dump((long)(char)x);   // expected-warning{{(long) (reg_$0<char x>)}}
  clang_analyzer_dump((long)(short)x);  // expected-warning{{(long) (reg_$0<char x>)}}
  clang_analyzer_dump((long)(int)x);    // expected-warning{{(long) (reg_$0<char x>)}}
  clang_analyzer_dump((long)(long)x);   // expected-warning{{(long) (reg_$0<char x>)}}
  clang_analyzer_dump((long)(llong)x);  // expected-warning{{(long) (reg_$0<char x>)}}
  clang_analyzer_dump((long)(uchar)x);  // expected-warning{{(long) ((unsigned char) (reg_$0<char x>))}}
  clang_analyzer_dump((long)(ushort)x); // expected-warning{{(long) ((unsigned short) (reg_$0<char x>))}}
  clang_analyzer_dump((long)(uint)x);   // expected-warning{{(long) ((unsigned int) (reg_$0<char x>))}}
  clang_analyzer_dump((long)(ulong)x);  // expected-warning{{(long) (reg_$0<char x>)}}
  clang_analyzer_dump((long)(ullong)x); // expected-warning{{(long) (reg_$0<char x>)}}

  clang_analyzer_dump((llong)(schar)x);  // expected-warning{{(long long) (reg_$0<char x>)}}
  clang_analyzer_dump((llong)(char)x);   // expected-warning{{(long long) (reg_$0<char x>)}}
  clang_analyzer_dump((llong)(short)x);  // expected-warning{{(long long) (reg_$0<char x>)}}
  clang_analyzer_dump((llong)(int)x);    // expected-warning{{(long long) (reg_$0<char x>)}}
  clang_analyzer_dump((llong)(long)x);   // expected-warning{{(long long) (reg_$0<char x>)}}
  clang_analyzer_dump((llong)(llong)x);  // expected-warning{{(long long) (reg_$0<char x>)}}
  clang_analyzer_dump((llong)(uchar)x);  // expected-warning{{(long long) ((unsigned char) (reg_$0<char x>))}}
  clang_analyzer_dump((llong)(ushort)x); // expected-warning{{(long long) ((unsigned short) (reg_$0<char x>))}}
  clang_analyzer_dump((llong)(uint)x);   // expected-warning{{(long long) ((unsigned int) (reg_$0<char x>))}}
  clang_analyzer_dump((llong)(ulong)x);  // expected-warning{{(long long) (reg_$0<char x>)}}
  clang_analyzer_dump((llong)(ullong)x); // expected-warning{{(long long) (reg_$0<char x>)}}

  clang_analyzer_dump((uchar)(schar)x);  // expected-warning{{(unsigned char) (reg_$0<char x>)}}
  clang_analyzer_dump((uchar)(char)x);   // expected-warning{{(unsigned char) (reg_$0<char x>)}}
  clang_analyzer_dump((uchar)(short)x);  // expected-warning{{(unsigned char) (reg_$0<char x>)}}
  clang_analyzer_dump((uchar)(int)x);    // expected-warning{{(unsigned char) (reg_$0<char x>)}}
  clang_analyzer_dump((uchar)(long)x);   // expected-warning{{(unsigned char) (reg_$0<char x>)}}
  clang_analyzer_dump((uchar)(llong)x);  // expected-warning{{(unsigned char) (reg_$0<char x>)}}
  clang_analyzer_dump((uchar)(uchar)x);  // expected-warning{{(unsigned char) (reg_$0<char x>)}}
  clang_analyzer_dump((uchar)(ushort)x); // expected-warning{{(unsigned char) (reg_$0<char x>)}}
  clang_analyzer_dump((uchar)(uint)x);   // expected-warning{{(unsigned char) (reg_$0<char x>)}}
  clang_analyzer_dump((uchar)(ulong)x);  // expected-warning{{(unsigned char) (reg_$0<char x>)}}
  clang_analyzer_dump((uchar)(ullong)x); // expected-warning{{(unsigned char) (reg_$0<char x>)}}

  clang_analyzer_dump((ushort)(schar)x);  // expected-warning{{(unsigned short) (reg_$0<char x>)}}
  clang_analyzer_dump((ushort)(char)x);   // expected-warning{{(unsigned short) (reg_$0<char x>)}}
  clang_analyzer_dump((ushort)(short)x);  // expected-warning{{(unsigned short) (reg_$0<char x>)}}
  clang_analyzer_dump((ushort)(int)x);    // expected-warning{{(unsigned short) (reg_$0<char x>)}}
  clang_analyzer_dump((ushort)(long)x);   // expected-warning{{(unsigned short) (reg_$0<char x>)}}
  clang_analyzer_dump((ushort)(llong)x);  // expected-warning{{(unsigned short) (reg_$0<char x>)}}
  clang_analyzer_dump((ushort)(uchar)x);  // expected-warning{{(unsigned short) ((unsigned char) (reg_$0<char x>))}}
  clang_analyzer_dump((ushort)(ushort)x); // expected-warning{{(unsigned short) (reg_$0<char x>)}}
  clang_analyzer_dump((ushort)(uint)x);   // expected-warning{{(unsigned short) (reg_$0<char x>)}}
  clang_analyzer_dump((ushort)(ulong)x);  // expected-warning{{(unsigned short) (reg_$0<char x>)}}
  clang_analyzer_dump((ushort)(ullong)x); // expected-warning{{(unsigned short) (reg_$0<char x>)}}

  clang_analyzer_dump((uint)(schar)x);  // expected-warning{{(unsigned int) (reg_$0<char x>)}}
  clang_analyzer_dump((uint)(char)x);   // expected-warning{{(unsigned int) (reg_$0<char x>)}}
  clang_analyzer_dump((uint)(short)x);  // expected-warning{{(unsigned int) (reg_$0<char x>)}}
  clang_analyzer_dump((uint)(int)x);    // expected-warning{{(unsigned int) (reg_$0<char x>)}}
  clang_analyzer_dump((uint)(long)x);   // expected-warning{{(unsigned int) (reg_$0<char x>)}}
  clang_analyzer_dump((uint)(llong)x);  // expected-warning{{(unsigned int) (reg_$0<char x>)}}
  clang_analyzer_dump((uint)(uchar)x);  // expected-warning{{(unsigned int) ((unsigned char) (reg_$0<char x>))}}
  clang_analyzer_dump((uint)(ushort)x); // expected-warning{{(unsigned int) ((unsigned short) (reg_$0<char x>))}}
  clang_analyzer_dump((uint)(uint)x);   // expected-warning{{(unsigned int) (reg_$0<char x>)}}
  clang_analyzer_dump((uint)(ulong)x);  // expected-warning{{(unsigned int) (reg_$0<char x>)}}
  clang_analyzer_dump((uint)(ullong)x); // expected-warning{{(unsigned int) (reg_$0<char x>)}}

  clang_analyzer_dump((ulong)(schar)x);  // expected-warning{{(unsigned long) (reg_$0<char x>)}}
  clang_analyzer_dump((ulong)(char)x);   // expected-warning{{(unsigned long) (reg_$0<char x>)}}
  clang_analyzer_dump((ulong)(short)x);  // expected-warning{{(unsigned long) (reg_$0<char x>)}}
  clang_analyzer_dump((ulong)(int)x);    // expected-warning{{(unsigned long) (reg_$0<char x>)}}
  clang_analyzer_dump((ulong)(long)x);   // expected-warning{{(unsigned long) (reg_$0<char x>)}}
  clang_analyzer_dump((ulong)(llong)x);  // expected-warning{{(unsigned long) (reg_$0<char x>)}}
  clang_analyzer_dump((ulong)(uchar)x);  // expected-warning{{(unsigned long) ((unsigned char) (reg_$0<char x>))}}
  clang_analyzer_dump((ulong)(ushort)x); // expected-warning{{(unsigned long) ((unsigned short) (reg_$0<char x>))}}
  clang_analyzer_dump((ulong)(uint)x);   // expected-warning{{(unsigned long) ((unsigned int) (reg_$0<char x>))}}
  clang_analyzer_dump((ulong)(ulong)x);  // expected-warning{{(unsigned long) (reg_$0<char x>)}}
  clang_analyzer_dump((ulong)(ullong)x); // expected-warning{{(unsigned long) (reg_$0<char x>)}}

  clang_analyzer_dump((ullong)(schar)x);  // expected-warning{{(unsigned long long) (reg_$0<char x>)}}
  clang_analyzer_dump((ullong)(char)x);   // expected-warning{{(unsigned long long) (reg_$0<char x>)}}
  clang_analyzer_dump((ullong)(short)x);  // expected-warning{{(unsigned long long) (reg_$0<char x>)}}
  clang_analyzer_dump((ullong)(int)x);    // expected-warning{{(unsigned long long) (reg_$0<char x>)}}
  clang_analyzer_dump((ullong)(long)x);   // expected-warning{{(unsigned long long) (reg_$0<char x>)}}
  clang_analyzer_dump((ullong)(llong)x);  // expected-warning{{(unsigned long long) (reg_$0<char x>)}}
  clang_analyzer_dump((ullong)(uchar)x);  // expected-warning{{(unsigned long long) ((unsigned char) (reg_$0<char x>))}}
  clang_analyzer_dump((ullong)(ushort)x); // expected-warning{{(unsigned long long) ((unsigned short) (reg_$0<char x>))}}
  clang_analyzer_dump((ullong)(uint)x);   // expected-warning{{(unsigned long long) ((unsigned int) (reg_$0<char x>))}}
  clang_analyzer_dump((ullong)(ulong)x);  // expected-warning{{(unsigned long long) (reg_$0<char x>)}}
  clang_analyzer_dump((ullong)(ullong)x); // expected-warning{{(unsigned long long) (reg_$0<char x>)}}
}

void test_short(short x) {
  clang_analyzer_dump(x); // expected-warning{{reg_$0<short x>}}

  clang_analyzer_dump((schar)x);  // expected-warning{{(signed char) (reg_$0<short x>)}}
  clang_analyzer_dump((char)x);   // expected-warning{{(char) (reg_$0<short x>)}}
  clang_analyzer_dump((short)x);  // expected-warning{{reg_$0<short x>}}
  clang_analyzer_dump((int)x);    // expected-warning{{(int) (reg_$0<short x>)}}
  clang_analyzer_dump((long)x);   // expected-warning{{(long) (reg_$0<short x>)}}
  clang_analyzer_dump((llong)x);  // expected-warning{{(long long) (reg_$0<short x>)}}
  clang_analyzer_dump((uchar)x);  // expected-warning{{(unsigned char) (reg_$0<short x>)}}
  clang_analyzer_dump((ushort)x); // expected-warning{{(unsigned short) (reg_$0<short x>)}}
  clang_analyzer_dump((uint)x);   // expected-warning{{(unsigned int) (reg_$0<short x>)}}
  clang_analyzer_dump((ulong)x);  // expected-warning{{(unsigned long) (reg_$0<short x>)}}
  clang_analyzer_dump((ullong)x); // expected-warning{{(unsigned long long) (reg_$0<short x>)}}

  clang_analyzer_dump((schar)(schar)x);  // expected-warning{{(signed char) (reg_$0<short x>)}}
  clang_analyzer_dump((schar)(char)x);   // expected-warning{{(signed char) (reg_$0<short x>)}}
  clang_analyzer_dump((schar)(short)x);  // expected-warning{{(signed char) (reg_$0<short x>)}}
  clang_analyzer_dump((schar)(int)x);    // expected-warning{{(signed char) (reg_$0<short x>)}}
  clang_analyzer_dump((schar)(long)x);   // expected-warning{{(signed char) (reg_$0<short x>)}}
  clang_analyzer_dump((schar)(llong)x);  // expected-warning{{(signed char) (reg_$0<short x>)}}
  clang_analyzer_dump((schar)(uchar)x);  // expected-warning{{(signed char) (reg_$0<short x>)}}
  clang_analyzer_dump((schar)(ushort)x); // expected-warning{{(signed char) (reg_$0<short x>)}}
  clang_analyzer_dump((schar)(uint)x);   // expected-warning{{(signed char) (reg_$0<short x>)}}
  clang_analyzer_dump((schar)(ulong)x);  // expected-warning{{(signed char) (reg_$0<short x>)}}
  clang_analyzer_dump((schar)(ullong)x); // expected-warning{{(signed char) (reg_$0<short x>)}}

  clang_analyzer_dump((char)(schar)x);  // expected-warning{{(char) (reg_$0<short x>)}}
  clang_analyzer_dump((char)(char)x);   // expected-warning{{(char) (reg_$0<short x>)}}
  clang_analyzer_dump((char)(short)x);  // expected-warning{{(char) (reg_$0<short x>)}}
  clang_analyzer_dump((char)(int)x);    // expected-warning{{(char) (reg_$0<short x>)}}
  clang_analyzer_dump((char)(long)x);   // expected-warning{{(char) (reg_$0<short x>)}}
  clang_analyzer_dump((char)(llong)x);  // expected-warning{{(char) (reg_$0<short x>)}}
  clang_analyzer_dump((char)(uchar)x);  // expected-warning{{(char) (reg_$0<short x>)}}
  clang_analyzer_dump((char)(ushort)x); // expected-warning{{(char) (reg_$0<short x>)}}
  clang_analyzer_dump((char)(uint)x);   // expected-warning{{(char) (reg_$0<short x>)}}
  clang_analyzer_dump((char)(ulong)x);  // expected-warning{{(char) (reg_$0<short x>)}}
  clang_analyzer_dump((char)(ullong)x); // expected-warning{{(char) (reg_$0<short x>)}}

  clang_analyzer_dump((short)(schar)x);  // expected-warning{{reg_$0<short x>}}
  clang_analyzer_dump((short)(char)x);   // expected-warning{{(short) ((char) (reg_$0<short x>))}}
  clang_analyzer_dump((short)(short)x);  // expected-warning{{reg_$0<short x>}}
  clang_analyzer_dump((short)(int)x);    // expected-warning{{reg_$0<short x>}}
  clang_analyzer_dump((short)(long)x);   // expected-warning{{reg_$0<short x>}}
  clang_analyzer_dump((short)(llong)x);  // expected-warning{{reg_$0<short x>}}
  clang_analyzer_dump((short)(uchar)x);  // expected-warning{{(short) ((unsigned char) (reg_$0<short x>))}}
  clang_analyzer_dump((short)(ushort)x); // expected-warning{{reg_$0<short x>}}
  clang_analyzer_dump((short)(uint)x);   // expected-warning{{reg_$0<short x>}}
  clang_analyzer_dump((short)(ulong)x);  // expected-warning{{reg_$0<short x>}}
  clang_analyzer_dump((short)(ullong)x); // expected-warning{{reg_$0<short x>}}

  clang_analyzer_dump((int)(schar)x);  // expected-warning{{(int) ((signed char) (reg_$0<short x>))}}
  clang_analyzer_dump((int)(char)x);   // expected-warning{{(int) ((char) (reg_$0<short x>))}}
  clang_analyzer_dump((int)(short)x);  // expected-warning{{(int) (reg_$0<short x>)}}
  clang_analyzer_dump((int)(int)x);    // expected-warning{{(int) (reg_$0<short x>)}}
  clang_analyzer_dump((int)(long)x);   // expected-warning{{(int) (reg_$0<short x>)}}
  clang_analyzer_dump((int)(llong)x);  // expected-warning{{(int) (reg_$0<short x>)}}
  clang_analyzer_dump((int)(uchar)x);  // expected-warning{{(int) ((unsigned char) (reg_$0<short x>))}}
  clang_analyzer_dump((int)(ushort)x); // expected-warning{{(int) ((unsigned short) (reg_$0<short x>))}}
  clang_analyzer_dump((int)(uint)x);   // expected-warning{{(int) (reg_$0<short x>)}}
  clang_analyzer_dump((int)(ulong)x);  // expected-warning{{(int) (reg_$0<short x>)}}
  clang_analyzer_dump((int)(ullong)x); // expected-warning{{(int) (reg_$0<short x>)}}

  clang_analyzer_dump((long)(schar)x);  // expected-warning{{(long) ((signed char) (reg_$0<short x>))}}
  clang_analyzer_dump((long)(char)x);   // expected-warning{{(long) ((char) (reg_$0<short x>))}}
  clang_analyzer_dump((long)(short)x);  // expected-warning{{(long) (reg_$0<short x>)}}
  clang_analyzer_dump((long)(int)x);    // expected-warning{{(long) (reg_$0<short x>)}}
  clang_analyzer_dump((long)(long)x);   // expected-warning{{(long) (reg_$0<short x>)}}
  clang_analyzer_dump((long)(llong)x);  // expected-warning{{(long) (reg_$0<short x>)}}
  clang_analyzer_dump((long)(uchar)x);  // expected-warning{{(long) ((unsigned char) (reg_$0<short x>))}}
  clang_analyzer_dump((long)(ushort)x); // expected-warning{{(long) ((unsigned short) (reg_$0<short x>))}}
  clang_analyzer_dump((long)(uint)x);   // expected-warning{{(long) ((unsigned int) (reg_$0<short x>))}}
  clang_analyzer_dump((long)(ulong)x);  // expected-warning{{(long) (reg_$0<short x>)}}
  clang_analyzer_dump((long)(ullong)x); // expected-warning{{(long) (reg_$0<short x>)}}

  clang_analyzer_dump((llong)(schar)x);  // expected-warning{{(long long) ((signed char) (reg_$0<short x>))}}
  clang_analyzer_dump((llong)(char)x);   // expected-warning{{(long long) ((char) (reg_$0<short x>))}}
  clang_analyzer_dump((llong)(short)x);  // expected-warning{{(long long) (reg_$0<short x>)}}
  clang_analyzer_dump((llong)(int)x);    // expected-warning{{(long long) (reg_$0<short x>)}}
  clang_analyzer_dump((llong)(long)x);   // expected-warning{{(long long) (reg_$0<short x>)}}
  clang_analyzer_dump((llong)(llong)x);  // expected-warning{{(long long) (reg_$0<short x>)}}
  clang_analyzer_dump((llong)(uchar)x);  // expected-warning{{(long long) ((unsigned char) (reg_$0<short x>))}}
  clang_analyzer_dump((llong)(ushort)x); // expected-warning{{(long long) ((unsigned short) (reg_$0<short x>))}}
  clang_analyzer_dump((llong)(uint)x);   // expected-warning{{(long long) ((unsigned int) (reg_$0<short x>))}}
  clang_analyzer_dump((llong)(ulong)x);  // expected-warning{{(long long) (reg_$0<short x>)}}
  clang_analyzer_dump((llong)(ullong)x); // expected-warning{{(long long) (reg_$0<short x>)}}

  clang_analyzer_dump((uchar)(schar)x);  // expected-warning{{(unsigned char) (reg_$0<short x>)}}
  clang_analyzer_dump((uchar)(char)x);   // expected-warning{{(unsigned char) (reg_$0<short x>)}}
  clang_analyzer_dump((uchar)(short)x);  // expected-warning{{(unsigned char) (reg_$0<short x>)}}
  clang_analyzer_dump((uchar)(int)x);    // expected-warning{{(unsigned char) (reg_$0<short x>)}}
  clang_analyzer_dump((uchar)(long)x);   // expected-warning{{(unsigned char) (reg_$0<short x>)}}
  clang_analyzer_dump((uchar)(llong)x);  // expected-warning{{(unsigned char) (reg_$0<short x>)}}
  clang_analyzer_dump((uchar)(uchar)x);  // expected-warning{{(unsigned char) (reg_$0<short x>)}}
  clang_analyzer_dump((uchar)(ushort)x); // expected-warning{{(unsigned char) (reg_$0<short x>)}}
  clang_analyzer_dump((uchar)(uint)x);   // expected-warning{{(unsigned char) (reg_$0<short x>)}}
  clang_analyzer_dump((uchar)(ulong)x);  // expected-warning{{(unsigned char) (reg_$0<short x>)}}
  clang_analyzer_dump((uchar)(ullong)x); // expected-warning{{(unsigned char) (reg_$0<short x>)}}

  clang_analyzer_dump((ushort)(schar)x);  // expected-warning{{(unsigned short) ((signed char) (reg_$0<short x>))}}
  clang_analyzer_dump((ushort)(char)x);   // expected-warning{{(unsigned short) ((char) (reg_$0<short x>))}}
  clang_analyzer_dump((ushort)(short)x);  // expected-warning{{(unsigned short) (reg_$0<short x>)}}
  clang_analyzer_dump((ushort)(int)x);    // expected-warning{{(unsigned short) (reg_$0<short x>)}}
  clang_analyzer_dump((ushort)(long)x);   // expected-warning{{(unsigned short) (reg_$0<short x>)}}
  clang_analyzer_dump((ushort)(llong)x);  // expected-warning{{(unsigned short) (reg_$0<short x>)}}
  clang_analyzer_dump((ushort)(uchar)x);  // expected-warning{{(unsigned short) ((unsigned char) (reg_$0<short x>))}}
  clang_analyzer_dump((ushort)(ushort)x); // expected-warning{{(unsigned short) (reg_$0<short x>)}}
  clang_analyzer_dump((ushort)(uint)x);   // expected-warning{{(unsigned short) (reg_$0<short x>)}}
  clang_analyzer_dump((ushort)(ulong)x);  // expected-warning{{(unsigned short) (reg_$0<short x>)}}
  clang_analyzer_dump((ushort)(ullong)x); // expected-warning{{(unsigned short) (reg_$0<short x>)}}

  clang_analyzer_dump((uint)(schar)x);  // expected-warning{{(unsigned int) ((signed char) (reg_$0<short x>))}}
  clang_analyzer_dump((uint)(char)x);   // expected-warning{{(unsigned int) ((char) (reg_$0<short x>))}}
  clang_analyzer_dump((uint)(short)x);  // expected-warning{{(unsigned int) (reg_$0<short x>)}}
  clang_analyzer_dump((uint)(int)x);    // expected-warning{{(unsigned int) (reg_$0<short x>)}}
  clang_analyzer_dump((uint)(long)x);   // expected-warning{{(unsigned int) (reg_$0<short x>)}}
  clang_analyzer_dump((uint)(llong)x);  // expected-warning{{(unsigned int) (reg_$0<short x>)}}
  clang_analyzer_dump((uint)(uchar)x);  // expected-warning{{(unsigned int) ((unsigned char) (reg_$0<short x>))}}
  clang_analyzer_dump((uint)(ushort)x); // expected-warning{{(unsigned int) ((unsigned short) (reg_$0<short x>))}}
  clang_analyzer_dump((uint)(uint)x);   // expected-warning{{(unsigned int) (reg_$0<short x>)}}
  clang_analyzer_dump((uint)(ulong)x);  // expected-warning{{(unsigned int) (reg_$0<short x>)}}
  clang_analyzer_dump((uint)(ullong)x); // expected-warning{{(unsigned int) (reg_$0<short x>)}}

  clang_analyzer_dump((ulong)(schar)x);  // expected-warning{{(unsigned long) ((signed char) (reg_$0<short x>))}}
  clang_analyzer_dump((ulong)(char)x);   // expected-warning{{(unsigned long) ((char) (reg_$0<short x>))}}
  clang_analyzer_dump((ulong)(short)x);  // expected-warning{{(unsigned long) (reg_$0<short x>)}}
  clang_analyzer_dump((ulong)(int)x);    // expected-warning{{(unsigned long) (reg_$0<short x>)}}
  clang_analyzer_dump((ulong)(long)x);   // expected-warning{{(unsigned long) (reg_$0<short x>)}}
  clang_analyzer_dump((ulong)(llong)x);  // expected-warning{{(unsigned long) (reg_$0<short x>)}}
  clang_analyzer_dump((ulong)(uchar)x);  // expected-warning{{(unsigned long) ((unsigned char) (reg_$0<short x>))}}
  clang_analyzer_dump((ulong)(ushort)x); // expected-warning{{(unsigned long) ((unsigned short) (reg_$0<short x>))}}
  clang_analyzer_dump((ulong)(uint)x);   // expected-warning{{(unsigned long) ((unsigned int) (reg_$0<short x>))}}
  clang_analyzer_dump((ulong)(ulong)x);  // expected-warning{{(unsigned long) (reg_$0<short x>)}}
  clang_analyzer_dump((ulong)(ullong)x); // expected-warning{{(unsigned long) (reg_$0<short x>)}}

  clang_analyzer_dump((ullong)(schar)x);  // expected-warning{{(unsigned long long) ((signed char) (reg_$0<short x>))}}
  clang_analyzer_dump((ullong)(char)x);   // expected-warning{{(unsigned long long) ((char) (reg_$0<short x>))}}
  clang_analyzer_dump((ullong)(short)x);  // expected-warning{{(unsigned long long) (reg_$0<short x>)}}
  clang_analyzer_dump((ullong)(int)x);    // expected-warning{{(unsigned long long) (reg_$0<short x>)}}
  clang_analyzer_dump((ullong)(long)x);   // expected-warning{{(unsigned long long) (reg_$0<short x>)}}
  clang_analyzer_dump((ullong)(llong)x);  // expected-warning{{(unsigned long long) (reg_$0<short x>)}}
  clang_analyzer_dump((ullong)(uchar)x);  // expected-warning{{(unsigned long long) ((unsigned char) (reg_$0<short x>))}}
  clang_analyzer_dump((ullong)(ushort)x); // expected-warning{{(unsigned long long) ((unsigned short) (reg_$0<short x>))}}
  clang_analyzer_dump((ullong)(uint)x);   // expected-warning{{(unsigned long long) ((unsigned int) (reg_$0<short x>))}}
  clang_analyzer_dump((ullong)(ulong)x);  // expected-warning{{(unsigned long long) (reg_$0<short x>)}}
  clang_analyzer_dump((ullong)(ullong)x); // expected-warning{{(unsigned long long) (reg_$0<short x>)}}
}

void test_int(int x) {
  clang_analyzer_dump(x); // expected-warning{{reg_$0<int x>}}

  clang_analyzer_dump((schar)x);  // expected-warning{{(signed char) (reg_$0<int x>)}}
  clang_analyzer_dump((char)x);   // expected-warning{{(char) (reg_$0<int x>)}}
  clang_analyzer_dump((short)x);  // expected-warning{{(short) (reg_$0<int x>)}}
  clang_analyzer_dump((int)x);    // expected-warning{{reg_$0<int x>}}
  clang_analyzer_dump((long)x);   // expected-warning{{(long) (reg_$0<int x>)}}
  clang_analyzer_dump((llong)x);  // expected-warning{{(long long) (reg_$0<int x>)}}
  clang_analyzer_dump((uchar)x);  // expected-warning{{(unsigned char) (reg_$0<int x>)}}
  clang_analyzer_dump((ushort)x); // expected-warning{{(unsigned short) (reg_$0<int x>)}}
  clang_analyzer_dump((uint)x);   // expected-warning{{(unsigned int) (reg_$0<int x>)}}
  clang_analyzer_dump((ulong)x);  // expected-warning{{(unsigned long) (reg_$0<int x>)}}
  clang_analyzer_dump((ullong)x); // expected-warning{{(unsigned long long) (reg_$0<int x>)}}

  clang_analyzer_dump((schar)(schar)x);  // expected-warning{{(signed char) (reg_$0<int x>)}}
  clang_analyzer_dump((schar)(char)x);   // expected-warning{{(signed char) (reg_$0<int x>)}}
  clang_analyzer_dump((schar)(short)x);  // expected-warning{{(signed char) (reg_$0<int x>)}}
  clang_analyzer_dump((schar)(int)x);    // expected-warning{{(signed char) (reg_$0<int x>)}}
  clang_analyzer_dump((schar)(long)x);   // expected-warning{{(signed char) (reg_$0<int x>)}}
  clang_analyzer_dump((schar)(llong)x);  // expected-warning{{(signed char) (reg_$0<int x>)}}
  clang_analyzer_dump((schar)(uchar)x);  // expected-warning{{(signed char) (reg_$0<int x>)}}
  clang_analyzer_dump((schar)(ushort)x); // expected-warning{{(signed char) (reg_$0<int x>)}}
  clang_analyzer_dump((schar)(uint)x);   // expected-warning{{(signed char) (reg_$0<int x>)}}
  clang_analyzer_dump((schar)(ulong)x);  // expected-warning{{(signed char) (reg_$0<int x>)}}
  clang_analyzer_dump((schar)(ullong)x); // expected-warning{{(signed char) (reg_$0<int x>)}}

  clang_analyzer_dump((char)(schar)x);  // expected-warning{{(char) (reg_$0<int x>)}}
  clang_analyzer_dump((char)(char)x);   // expected-warning{{(char) (reg_$0<int x>)}}
  clang_analyzer_dump((char)(short)x);  // expected-warning{{(char) (reg_$0<int x>)}}
  clang_analyzer_dump((char)(int)x);    // expected-warning{{(char) (reg_$0<int x>)}}
  clang_analyzer_dump((char)(long)x);   // expected-warning{{(char) (reg_$0<int x>)}}
  clang_analyzer_dump((char)(llong)x);  // expected-warning{{(char) (reg_$0<int x>)}}
  clang_analyzer_dump((char)(uchar)x);  // expected-warning{{(char) (reg_$0<int x>)}}
  clang_analyzer_dump((char)(ushort)x); // expected-warning{{(char) (reg_$0<int x>)}}
  clang_analyzer_dump((char)(uint)x);   // expected-warning{{(char) (reg_$0<int x>)}}
  clang_analyzer_dump((char)(ulong)x);  // expected-warning{{(char) (reg_$0<int x>)}}
  clang_analyzer_dump((char)(ullong)x); // expected-warning{{(char) (reg_$0<int x>)}}

  clang_analyzer_dump((short)(schar)x);  // expected-warning{{(short) ((signed char) (reg_$0<int x>))}}
  clang_analyzer_dump((short)(char)x);   // expected-warning{{(short) ((char) (reg_$0<int x>))}}
  clang_analyzer_dump((short)(short)x);  // expected-warning{{(short) (reg_$0<int x>)}}
  clang_analyzer_dump((short)(int)x);    // expected-warning{{(short) (reg_$0<int x>)}}
  clang_analyzer_dump((short)(long)x);   // expected-warning{{(short) (reg_$0<int x>)}}
  clang_analyzer_dump((short)(llong)x);  // expected-warning{{(short) (reg_$0<int x>)}}
  clang_analyzer_dump((short)(uchar)x);  // expected-warning{{(short) ((unsigned char) (reg_$0<int x>))}}
  clang_analyzer_dump((short)(ushort)x); // expected-warning{{(short) (reg_$0<int x>)}}
  clang_analyzer_dump((short)(uint)x);   // expected-warning{{(short) (reg_$0<int x>)}}
  clang_analyzer_dump((short)(ulong)x);  // expected-warning{{(short) (reg_$0<int x>)}}
  clang_analyzer_dump((short)(ullong)x); // expected-warning{{(short) (reg_$0<int x>)}}

  clang_analyzer_dump((int)(schar)x);  // expected-warning{{reg_$0<int x>}}
  clang_analyzer_dump((int)(char)x);   // expected-warning{{(int) ((char) (reg_$0<int x>))}}
  clang_analyzer_dump((int)(short)x);  // expected-warning{{(int) ((short) (reg_$0<int x>))}}
  clang_analyzer_dump((int)(int)x);    // expected-warning{{reg_$0<int x>}}
  clang_analyzer_dump((int)(long)x);   // expected-warning{{reg_$0<int x>}}
  clang_analyzer_dump((int)(llong)x);  // expected-warning{{reg_$0<int x>}}
  clang_analyzer_dump((int)(uchar)x);  // expected-warning{{(int) ((unsigned char) (reg_$0<int x>))}}
  clang_analyzer_dump((int)(ushort)x); // expected-warning{{(int) ((unsigned short) (reg_$0<int x>))}}
  clang_analyzer_dump((int)(uint)x);   // expected-warning{{reg_$0<int x>}}
  clang_analyzer_dump((int)(ulong)x);  // expected-warning{{reg_$0<int x>}}
  clang_analyzer_dump((int)(ullong)x); // expected-warning{{reg_$0<int x>}}

  clang_analyzer_dump((long)(schar)x);  // expected-warning{{(long) ((signed char) (reg_$0<int x>))}}
  clang_analyzer_dump((long)(char)x);   // expected-warning{{(long) ((char) (reg_$0<int x>))}}
  clang_analyzer_dump((long)(short)x);  // expected-warning{{(long) ((short) (reg_$0<int x>))}}
  clang_analyzer_dump((long)(int)x);    // expected-warning{{(long) (reg_$0<int x>)}}
  clang_analyzer_dump((long)(long)x);   // expected-warning{{(long) (reg_$0<int x>)}}
  clang_analyzer_dump((long)(llong)x);  // expected-warning{{(long) (reg_$0<int x>)}}
  clang_analyzer_dump((long)(uchar)x);  // expected-warning{{(long) ((unsigned char) (reg_$0<int x>))}}
  clang_analyzer_dump((long)(ushort)x); // expected-warning{{(long) ((unsigned short) (reg_$0<int x>))}}
  clang_analyzer_dump((long)(uint)x);   // expected-warning{{(long) ((unsigned int) (reg_$0<int x>))}}
  clang_analyzer_dump((long)(ulong)x);  // expected-warning{{(long) (reg_$0<int x>)}}
  clang_analyzer_dump((long)(ullong)x); // expected-warning{{(long) (reg_$0<int x>)}}

  clang_analyzer_dump((llong)(schar)x);  // expected-warning{{(long long) ((signed char) (reg_$0<int x>))}}
  clang_analyzer_dump((llong)(char)x);   // expected-warning{{(long long) ((char) (reg_$0<int x>))}}
  clang_analyzer_dump((llong)(short)x);  // expected-warning{{(long long) ((short) (reg_$0<int x>))}}
  clang_analyzer_dump((llong)(int)x);    // expected-warning{{(long long) (reg_$0<int x>)}}
  clang_analyzer_dump((llong)(long)x);   // expected-warning{{(long long) (reg_$0<int x>)}}
  clang_analyzer_dump((llong)(llong)x);  // expected-warning{{(long long) (reg_$0<int x>)}}
  clang_analyzer_dump((llong)(uchar)x);  // expected-warning{{(long long) ((unsigned char) (reg_$0<int x>))}}
  clang_analyzer_dump((llong)(ushort)x); // expected-warning{{(long long) ((unsigned short) (reg_$0<int x>))}}
  clang_analyzer_dump((llong)(uint)x);   // expected-warning{{(long long) ((unsigned int) (reg_$0<int x>))}}
  clang_analyzer_dump((llong)(ulong)x);  // expected-warning{{(long long) (reg_$0<int x>)}}
  clang_analyzer_dump((llong)(ullong)x); // expected-warning{{(long long) (reg_$0<int x>)}}

  clang_analyzer_dump((uchar)(schar)x);  // expected-warning{{(unsigned char) (reg_$0<int x>)}}
  clang_analyzer_dump((uchar)(char)x);   // expected-warning{{(unsigned char) (reg_$0<int x>)}}
  clang_analyzer_dump((uchar)(short)x);  // expected-warning{{(unsigned char) (reg_$0<int x>)}}
  clang_analyzer_dump((uchar)(int)x);    // expected-warning{{(unsigned char) (reg_$0<int x>)}}
  clang_analyzer_dump((uchar)(long)x);   // expected-warning{{(unsigned char) (reg_$0<int x>)}}
  clang_analyzer_dump((uchar)(llong)x);  // expected-warning{{(unsigned char) (reg_$0<int x>)}}
  clang_analyzer_dump((uchar)(uchar)x);  // expected-warning{{(unsigned char) (reg_$0<int x>)}}
  clang_analyzer_dump((uchar)(ushort)x); // expected-warning{{(unsigned char) (reg_$0<int x>)}}
  clang_analyzer_dump((uchar)(uint)x);   // expected-warning{{(unsigned char) (reg_$0<int x>)}}
  clang_analyzer_dump((uchar)(ulong)x);  // expected-warning{{(unsigned char) (reg_$0<int x>)}}
  clang_analyzer_dump((uchar)(ullong)x); // expected-warning{{(unsigned char) (reg_$0<int x>)}}

  clang_analyzer_dump((ushort)(schar)x);  // expected-warning{{(unsigned short) ((signed char) (reg_$0<int x>))}}
  clang_analyzer_dump((ushort)(char)x);   // expected-warning{{(unsigned short) ((char) (reg_$0<int x>))}}
  clang_analyzer_dump((ushort)(short)x);  // expected-warning{{(unsigned short) (reg_$0<int x>)}}
  clang_analyzer_dump((ushort)(int)x);    // expected-warning{{(unsigned short) (reg_$0<int x>)}}
  clang_analyzer_dump((ushort)(long)x);   // expected-warning{{(unsigned short) (reg_$0<int x>)}}
  clang_analyzer_dump((ushort)(llong)x);  // expected-warning{{(unsigned short) (reg_$0<int x>)}}
  clang_analyzer_dump((ushort)(uchar)x);  // expected-warning{{(unsigned short) ((unsigned char) (reg_$0<int x>))}}
  clang_analyzer_dump((ushort)(ushort)x); // expected-warning{{(unsigned short) (reg_$0<int x>)}}
  clang_analyzer_dump((ushort)(uint)x);   // expected-warning{{(unsigned short) (reg_$0<int x>)}}
  clang_analyzer_dump((ushort)(ulong)x);  // expected-warning{{(unsigned short) (reg_$0<int x>)}}
  clang_analyzer_dump((ushort)(ullong)x); // expected-warning{{(unsigned short) (reg_$0<int x>)}}

  clang_analyzer_dump((uint)(schar)x);  // expected-warning{{(unsigned int) ((signed char) (reg_$0<int x>))}}
  clang_analyzer_dump((uint)(char)x);   // expected-warning{{(unsigned int) ((char) (reg_$0<int x>))}}
  clang_analyzer_dump((uint)(short)x);  // expected-warning{{(unsigned int) ((short) (reg_$0<int x>))}}
  clang_analyzer_dump((uint)(int)x);    // expected-warning{{(unsigned int) (reg_$0<int x>)}}
  clang_analyzer_dump((uint)(long)x);   // expected-warning{{(unsigned int) (reg_$0<int x>)}}
  clang_analyzer_dump((uint)(llong)x);  // expected-warning{{(unsigned int) (reg_$0<int x>)}}
  clang_analyzer_dump((uint)(uchar)x);  // expected-warning{{(unsigned int) ((unsigned char) (reg_$0<int x>))}}
  clang_analyzer_dump((uint)(ushort)x); // expected-warning{{(unsigned int) ((unsigned short) (reg_$0<int x>))}}
  clang_analyzer_dump((uint)(uint)x);   // expected-warning{{(unsigned int) (reg_$0<int x>)}}
  clang_analyzer_dump((uint)(ulong)x);  // expected-warning{{(unsigned int) (reg_$0<int x>)}}
  clang_analyzer_dump((uint)(ullong)x); // expected-warning{{(unsigned int) (reg_$0<int x>)}}

  clang_analyzer_dump((ulong)(schar)x);  // expected-warning{{(unsigned long) ((signed char) (reg_$0<int x>))}}
  clang_analyzer_dump((ulong)(char)x);   // expected-warning{{(unsigned long) ((char) (reg_$0<int x>))}}
  clang_analyzer_dump((ulong)(short)x);  // expected-warning{{(unsigned long) ((short) (reg_$0<int x>))}}
  clang_analyzer_dump((ulong)(int)x);    // expected-warning{{(unsigned long) (reg_$0<int x>)}}
  clang_analyzer_dump((ulong)(long)x);   // expected-warning{{(unsigned long) (reg_$0<int x>)}}
  clang_analyzer_dump((ulong)(llong)x);  // expected-warning{{(unsigned long) (reg_$0<int x>)}}
  clang_analyzer_dump((ulong)(uchar)x);  // expected-warning{{(unsigned long) ((unsigned char) (reg_$0<int x>))}}
  clang_analyzer_dump((ulong)(ushort)x); // expected-warning{{(unsigned long) ((unsigned short) (reg_$0<int x>))}}
  clang_analyzer_dump((ulong)(uint)x);   // expected-warning{{(unsigned long) ((unsigned int) (reg_$0<int x>))}}
  clang_analyzer_dump((ulong)(ulong)x);  // expected-warning{{(unsigned long) (reg_$0<int x>)}}
  clang_analyzer_dump((ulong)(ullong)x); // expected-warning{{(unsigned long) (reg_$0<int x>)}}

  clang_analyzer_dump((ullong)(schar)x);  // expected-warning{{(unsigned long long) ((signed char) (reg_$0<int x>))}}
  clang_analyzer_dump((ullong)(char)x);   // expected-warning{{(unsigned long long) ((char) (reg_$0<int x>))}}
  clang_analyzer_dump((ullong)(short)x);  // expected-warning{{(unsigned long long) ((short) (reg_$0<int x>))}}
  clang_analyzer_dump((ullong)(int)x);    // expected-warning{{(unsigned long long) (reg_$0<int x>)}}
  clang_analyzer_dump((ullong)(long)x);   // expected-warning{{(unsigned long long) (reg_$0<int x>)}}
  clang_analyzer_dump((ullong)(llong)x);  // expected-warning{{(unsigned long long) (reg_$0<int x>)}}
  clang_analyzer_dump((ullong)(uchar)x);  // expected-warning{{(unsigned long long) ((unsigned char) (reg_$0<int x>))}}
  clang_analyzer_dump((ullong)(ushort)x); // expected-warning{{(unsigned long long) ((unsigned short) (reg_$0<int x>))}}
  clang_analyzer_dump((ullong)(uint)x);   // expected-warning{{(unsigned long long) ((unsigned int) (reg_$0<int x>))}}
  clang_analyzer_dump((ullong)(ulong)x);  // expected-warning{{(unsigned long long) (reg_$0<int x>)}}
  clang_analyzer_dump((ullong)(ullong)x); // expected-warning{{(unsigned long long) (reg_$0<int x>)}}
}

void test_long(long x) {
  clang_analyzer_dump(x); // expected-warning{{reg_$0<long x>}}

  clang_analyzer_dump((schar)x);  // expected-warning{{(signed char) (reg_$0<long x>)}}
  clang_analyzer_dump((char)x);   // expected-warning{{(char) (reg_$0<long x>)}}
  clang_analyzer_dump((short)x);  // expected-warning{{(short) (reg_$0<long x>)}}
  clang_analyzer_dump((int)x);    // expected-warning{{(int) (reg_$0<long x>)}}
  clang_analyzer_dump((long)x);   // expected-warning{{reg_$0<long x>}}
  clang_analyzer_dump((llong)x);  // expected-warning{{(long long) (reg_$0<long x>)}}
  clang_analyzer_dump((uchar)x);  // expected-warning{{(unsigned char) (reg_$0<long x>)}}
  clang_analyzer_dump((ushort)x); // expected-warning{{(unsigned short) (reg_$0<long x>)}}
  clang_analyzer_dump((uint)x);   // expected-warning{{(unsigned int) (reg_$0<long x>)}}
  clang_analyzer_dump((ulong)x);  // expected-warning{{(unsigned long) (reg_$0<long x>)}}
  clang_analyzer_dump((ullong)x); // expected-warning{{(unsigned long long) (reg_$0<long x>)}}

  clang_analyzer_dump((schar)(schar)x);  // expected-warning{{(signed char) (reg_$0<long x>)}}
  clang_analyzer_dump((schar)(char)x);   // expected-warning{{(signed char) (reg_$0<long x>)}}
  clang_analyzer_dump((schar)(short)x);  // expected-warning{{(signed char) (reg_$0<long x>)}}
  clang_analyzer_dump((schar)(int)x);    // expected-warning{{(signed char) (reg_$0<long x>)}}
  clang_analyzer_dump((schar)(long)x);   // expected-warning{{(signed char) (reg_$0<long x>)}}
  clang_analyzer_dump((schar)(llong)x);  // expected-warning{{(signed char) (reg_$0<long x>)}}
  clang_analyzer_dump((schar)(uchar)x);  // expected-warning{{(signed char) (reg_$0<long x>)}}
  clang_analyzer_dump((schar)(ushort)x); // expected-warning{{(signed char) (reg_$0<long x>)}}
  clang_analyzer_dump((schar)(uint)x);   // expected-warning{{(signed char) (reg_$0<long x>)}}
  clang_analyzer_dump((schar)(ulong)x);  // expected-warning{{(signed char) (reg_$0<long x>)}}
  clang_analyzer_dump((schar)(ullong)x); // expected-warning{{(signed char) (reg_$0<long x>)}}

  clang_analyzer_dump((char)(schar)x);  // expected-warning{{(char) (reg_$0<long x>)}}
  clang_analyzer_dump((char)(char)x);   // expected-warning{{(char) (reg_$0<long x>)}}
  clang_analyzer_dump((char)(short)x);  // expected-warning{{(char) (reg_$0<long x>)}}
  clang_analyzer_dump((char)(int)x);    // expected-warning{{(char) (reg_$0<long x>)}}
  clang_analyzer_dump((char)(long)x);   // expected-warning{{(char) (reg_$0<long x>)}}
  clang_analyzer_dump((char)(llong)x);  // expected-warning{{(char) (reg_$0<long x>)}}
  clang_analyzer_dump((char)(uchar)x);  // expected-warning{{(char) (reg_$0<long x>)}}
  clang_analyzer_dump((char)(ushort)x); // expected-warning{{(char) (reg_$0<long x>)}}
  clang_analyzer_dump((char)(uint)x);   // expected-warning{{(char) (reg_$0<long x>)}}
  clang_analyzer_dump((char)(ulong)x);  // expected-warning{{(char) (reg_$0<long x>)}}
  clang_analyzer_dump((char)(ullong)x); // expected-warning{{(char) (reg_$0<long x>)}}

  clang_analyzer_dump((short)(schar)x);  // expected-warning{{(short) ((signed char) (reg_$0<long x>))}}
  clang_analyzer_dump((short)(char)x);   // expected-warning{{(short) ((char) (reg_$0<long x>))}}
  clang_analyzer_dump((short)(short)x);  // expected-warning{{(short) (reg_$0<long x>)}}
  clang_analyzer_dump((short)(int)x);    // expected-warning{{(short) (reg_$0<long x>)}}
  clang_analyzer_dump((short)(long)x);   // expected-warning{{(short) (reg_$0<long x>)}}
  clang_analyzer_dump((short)(llong)x);  // expected-warning{{(short) (reg_$0<long x>)}}
  clang_analyzer_dump((short)(uchar)x);  // expected-warning{{(short) ((unsigned char) (reg_$0<long x>))}}
  clang_analyzer_dump((short)(ushort)x); // expected-warning{{(short) (reg_$0<long x>)}}
  clang_analyzer_dump((short)(uint)x);   // expected-warning{{(short) (reg_$0<long x>)}}
  clang_analyzer_dump((short)(ulong)x);  // expected-warning{{(short) (reg_$0<long x>)}}
  clang_analyzer_dump((short)(ullong)x); // expected-warning{{(short) (reg_$0<long x>)}}

  clang_analyzer_dump((int)(schar)x);  // expected-warning{{(int) ((signed char) (reg_$0<long x>))}}
  clang_analyzer_dump((int)(char)x);   // expected-warning{{(int) ((char) (reg_$0<long x>))}}
  clang_analyzer_dump((int)(short)x);  // expected-warning{{(int) ((short) (reg_$0<long x>))}}
  clang_analyzer_dump((int)(int)x);    // expected-warning{{(int) (reg_$0<long x>)}}
  clang_analyzer_dump((int)(long)x);   // expected-warning{{(int) (reg_$0<long x>)}}
  clang_analyzer_dump((int)(llong)x);  // expected-warning{{(int) (reg_$0<long x>)}}
  clang_analyzer_dump((int)(uchar)x);  // expected-warning{{(int) ((unsigned char) (reg_$0<long x>))}}
  clang_analyzer_dump((int)(ushort)x); // expected-warning{{(int) ((unsigned short) (reg_$0<long x>))}}
  clang_analyzer_dump((int)(uint)x);   // expected-warning{{(int) (reg_$0<long x>)}}
  clang_analyzer_dump((int)(ulong)x);  // expected-warning{{(int) (reg_$0<long x>)}}
  clang_analyzer_dump((int)(ullong)x); // expected-warning{{(int) (reg_$0<long x>)}}

  clang_analyzer_dump((long)(schar)x);  // expected-warning{{reg_$0<long x>}}
  clang_analyzer_dump((long)(char)x);   // expected-warning{{(char) (reg_$0<long x>)}}
  clang_analyzer_dump((long)(short)x);  // expected-warning{{(short) (reg_$0<long x>)}}
  clang_analyzer_dump((long)(int)x);    // expected-warning{{reg_$0<long x>}}
  clang_analyzer_dump((long)(long)x);   // expected-warning{{reg_$0<long x>}}
  clang_analyzer_dump((long)(llong)x);  // expected-warning{{reg_$0<long x>}}
  clang_analyzer_dump((long)(uchar)x);  // expected-warning{{(unsigned char) (reg_$0<long x>)}}
  clang_analyzer_dump((long)(ushort)x); // expected-warning{{(unsigned short) (reg_$0<long x>)}}
  clang_analyzer_dump((long)(uint)x);   // expected-warning{{reg_$0<long x>}}
  clang_analyzer_dump((long)(ulong)x);  // expected-warning{{reg_$0<long x>}}
  clang_analyzer_dump((long)(ullong)x); // expected-warning{{reg_$0<long x>}}

  clang_analyzer_dump((llong)(schar)x);  // expected-warning{{(long long) ((signed char) (reg_$0<long x>))}}
  clang_analyzer_dump((llong)(char)x);   // expected-warning{{(long long) ((char) (reg_$0<long x>))}}
  clang_analyzer_dump((llong)(short)x);  // expected-warning{{(long long) ((short) (reg_$0<long x>))}}
  clang_analyzer_dump((llong)(int)x);    // expected-warning{{(long long) ((int) (reg_$0<long x>))}}
  clang_analyzer_dump((llong)(long)x);   // expected-warning{{(long long) (reg_$0<long x>)}}
  clang_analyzer_dump((llong)(llong)x);  // expected-warning{{(long long) (reg_$0<long x>)}}
  clang_analyzer_dump((llong)(uchar)x);  // expected-warning{{(long long) ((unsigned char) (reg_$0<long x>))}}
  clang_analyzer_dump((llong)(ushort)x); // expected-warning{{(long long) ((unsigned short) (reg_$0<long x>))}}
  clang_analyzer_dump((llong)(uint)x);   // expected-warning{{(long long) ((unsigned int) (reg_$0<long x>))}}
  clang_analyzer_dump((llong)(ulong)x);  // expected-warning{{(long long) (reg_$0<long x>)}}
  clang_analyzer_dump((llong)(ullong)x); // expected-warning{{(long long) (reg_$0<long x>)}}

  clang_analyzer_dump((uchar)(schar)x);  // expected-warning{{(unsigned char) (reg_$0<long x>)}}
  clang_analyzer_dump((uchar)(char)x);   // expected-warning{{(unsigned char) (reg_$0<long x>)}}
  clang_analyzer_dump((uchar)(short)x);  // expected-warning{{(unsigned char) (reg_$0<long x>)}}
  clang_analyzer_dump((uchar)(int)x);    // expected-warning{{(unsigned char) (reg_$0<long x>)}}
  clang_analyzer_dump((uchar)(long)x);   // expected-warning{{(unsigned char) (reg_$0<long x>)}}
  clang_analyzer_dump((uchar)(llong)x);  // expected-warning{{(unsigned char) (reg_$0<long x>)}}
  clang_analyzer_dump((uchar)(uchar)x);  // expected-warning{{(unsigned char) (reg_$0<long x>)}}
  clang_analyzer_dump((uchar)(ushort)x); // expected-warning{{(unsigned char) (reg_$0<long x>)}}
  clang_analyzer_dump((uchar)(uint)x);   // expected-warning{{(unsigned char) (reg_$0<long x>)}}
  clang_analyzer_dump((uchar)(ulong)x);  // expected-warning{{(unsigned char) (reg_$0<long x>)}}
  clang_analyzer_dump((uchar)(ullong)x); // expected-warning{{(unsigned char) (reg_$0<long x>)}}

  clang_analyzer_dump((ushort)(schar)x);  // expected-warning{{(unsigned short) ((signed char) (reg_$0<long x>))}}
  clang_analyzer_dump((ushort)(char)x);   // expected-warning{{(unsigned short) ((char) (reg_$0<long x>))}}
  clang_analyzer_dump((ushort)(short)x);  // expected-warning{{(unsigned short) (reg_$0<long x>)}}
  clang_analyzer_dump((ushort)(int)x);    // expected-warning{{(unsigned short) (reg_$0<long x>)}}
  clang_analyzer_dump((ushort)(long)x);   // expected-warning{{(unsigned short) (reg_$0<long x>)}}
  clang_analyzer_dump((ushort)(llong)x);  // expected-warning{{(unsigned short) (reg_$0<long x>)}}
  clang_analyzer_dump((ushort)(uchar)x);  // expected-warning{{(unsigned short) ((unsigned char) (reg_$0<long x>))}}
  clang_analyzer_dump((ushort)(ushort)x); // expected-warning{{(unsigned short) (reg_$0<long x>)}}
  clang_analyzer_dump((ushort)(uint)x);   // expected-warning{{(unsigned short) (reg_$0<long x>)}}
  clang_analyzer_dump((ushort)(ulong)x);  // expected-warning{{(unsigned short) (reg_$0<long x>)}}
  clang_analyzer_dump((ushort)(ullong)x); // expected-warning{{(unsigned short) (reg_$0<long x>)}}

  clang_analyzer_dump((uint)(schar)x);  // expected-warning{{(unsigned int) ((signed char) (reg_$0<long x>))}}
  clang_analyzer_dump((uint)(char)x);   // expected-warning{{(unsigned int) ((char) (reg_$0<long x>))}}
  clang_analyzer_dump((uint)(short)x);  // expected-warning{{(unsigned int) ((short) (reg_$0<long x>))}}
  clang_analyzer_dump((uint)(int)x);    // expected-warning{{(unsigned int) (reg_$0<long x>}}
  clang_analyzer_dump((uint)(long)x);   // expected-warning{{(unsigned int) (reg_$0<long x>}}
  clang_analyzer_dump((uint)(llong)x);  // expected-warning{{(unsigned int) (reg_$0<long x>}}
  clang_analyzer_dump((uint)(uchar)x);  // expected-warning{{(unsigned int) ((unsigned char) (reg_$0<long x>))}}
  clang_analyzer_dump((uint)(ushort)x); // expected-warning{{(unsigned int) ((unsigned short) (reg_$0<long x>))}}
  clang_analyzer_dump((uint)(uint)x);   // expected-warning{{(unsigned int) (reg_$0<long x>}}
  clang_analyzer_dump((uint)(ulong)x);  // expected-warning{{(unsigned int) (reg_$0<long x>}}
  clang_analyzer_dump((uint)(ullong)x); // expected-warning{{(unsigned int) (reg_$0<long x>}}

  clang_analyzer_dump((ulong)(schar)x);  // expected-warning{{(unsigned long) ((signed char) (reg_$0<long x>))}}
  clang_analyzer_dump((ulong)(char)x);   // expected-warning{{(unsigned long) ((char) (reg_$0<long x>))}}
  clang_analyzer_dump((ulong)(short)x);  // expected-warning{{(unsigned long) ((short) (reg_$0<long x>))}}
  clang_analyzer_dump((ulong)(int)x);    // expected-warning{{(unsigned long) ((int) (reg_$0<long x>))}}
  clang_analyzer_dump((ulong)(long)x);   // expected-warning{{(unsigned long) (reg_$0<long x>)}}
  clang_analyzer_dump((ulong)(llong)x);  // expected-warning{{(unsigned long) (reg_$0<long x>)}}
  clang_analyzer_dump((ulong)(uchar)x);  // expected-warning{{(unsigned long) ((unsigned char) (reg_$0<long x>))}}
  clang_analyzer_dump((ulong)(ushort)x); // expected-warning{{(unsigned long) ((unsigned short) (reg_$0<long x>))}}
  clang_analyzer_dump((ulong)(uint)x);   // expected-warning{{(unsigned long) ((unsigned int) (reg_$0<long x>))}}
  clang_analyzer_dump((ulong)(ulong)x);  // expected-warning{{(unsigned long) (reg_$0<long x>)}}
  clang_analyzer_dump((ulong)(ullong)x); // expected-warning{{(unsigned long) (reg_$0<long x>)}}

  clang_analyzer_dump((ullong)(schar)x);  // expected-warning{{(unsigned long long) ((signed char) (reg_$0<long x>))}}
  clang_analyzer_dump((ullong)(char)x);   // expected-warning{{(unsigned long long) ((char) (reg_$0<long x>))}}
  clang_analyzer_dump((ullong)(short)x);  // expected-warning{{(unsigned long long) ((short) (reg_$0<long x>))}}
  clang_analyzer_dump((ullong)(int)x);    // expected-warning{{(unsigned long long) ((int) (reg_$0<long x>))}}
  clang_analyzer_dump((ullong)(long)x);   // expected-warning{{(unsigned long long) (reg_$0<long x>)}}
  clang_analyzer_dump((ullong)(llong)x);  // expected-warning{{(unsigned long long) (reg_$0<long x>)}}
  clang_analyzer_dump((ullong)(uchar)x);  // expected-warning{{(unsigned long long) ((unsigned char) (reg_$0<long x>))}}
  clang_analyzer_dump((ullong)(ushort)x); // expected-warning{{(unsigned long long) ((unsigned short) (reg_$0<long x>))}}
  clang_analyzer_dump((ullong)(uint)x);   // expected-warning{{(unsigned long long) ((unsigned int) (reg_$0<long x>))}}
  clang_analyzer_dump((ullong)(ulong)x);  // expected-warning{{(unsigned long long) (reg_$0<long x>)}}
  clang_analyzer_dump((ullong)(ullong)x); // expected-warning{{(unsigned long long) (reg_$0<long x>)}}
}

void test_llong(llong x) {
  clang_analyzer_dump(x); // expected-warning{{reg_$0<llong x>}}

  clang_analyzer_dump((schar)x);  // expected-warning{{(signed char) (reg_$0<llong x>)}}
  clang_analyzer_dump((char)x);   // expected-warning{{(char) (reg_$0<llong x>)}}
  clang_analyzer_dump((short)x);  // expected-warning{{(short) (reg_$0<llong x>)}}
  clang_analyzer_dump((int)x);    // expected-warning{{(int) (reg_$0<llong x>)}}
  clang_analyzer_dump((long)x);   // expected-warning{{(long) (reg_$0<llong x>)}}
  clang_analyzer_dump((llong)x);  // expected-warning{{reg_$0<llong x>}}
  clang_analyzer_dump((uchar)x);  // expected-warning{{(unsigned char) (reg_$0<llong x>)}}
  clang_analyzer_dump((ushort)x); // expected-warning{{(unsigned short) (reg_$0<llong x>)}}
  clang_analyzer_dump((uint)x);   // expected-warning{{(unsigned int) (reg_$0<llong x>)}}
  clang_analyzer_dump((ulong)x);  // expected-warning{{(unsigned long) (reg_$0<llong x>)}}
  clang_analyzer_dump((ullong)x); // expected-warning{{(unsigned long long) (reg_$0<llong x>)}}

  clang_analyzer_dump((schar)(schar)x);  // expected-warning{{(signed char) (reg_$0<llong x>)}}
  clang_analyzer_dump((schar)(char)x);   // expected-warning{{(signed char) (reg_$0<llong x>)}}
  clang_analyzer_dump((schar)(short)x);  // expected-warning{{(signed char) (reg_$0<llong x>)}}
  clang_analyzer_dump((schar)(int)x);    // expected-warning{{(signed char) (reg_$0<llong x>)}}
  clang_analyzer_dump((schar)(long)x);   // expected-warning{{(signed char) (reg_$0<llong x>)}}
  clang_analyzer_dump((schar)(llong)x);  // expected-warning{{(signed char) (reg_$0<llong x>)}}
  clang_analyzer_dump((schar)(uchar)x);  // expected-warning{{(signed char) (reg_$0<llong x>)}}
  clang_analyzer_dump((schar)(ushort)x); // expected-warning{{(signed char) (reg_$0<llong x>)}}
  clang_analyzer_dump((schar)(uint)x);   // expected-warning{{(signed char) (reg_$0<llong x>)}}
  clang_analyzer_dump((schar)(ulong)x);  // expected-warning{{(signed char) (reg_$0<llong x>)}}
  clang_analyzer_dump((schar)(ullong)x); // expected-warning{{(signed char) (reg_$0<llong x>)}}

  clang_analyzer_dump((char)(schar)x);  // expected-warning{{(char) (reg_$0<llong x>)}}
  clang_analyzer_dump((char)(char)x);   // expected-warning{{(char) (reg_$0<llong x>)}}
  clang_analyzer_dump((char)(short)x);  // expected-warning{{(char) (reg_$0<llong x>)}}
  clang_analyzer_dump((char)(int)x);    // expected-warning{{(char) (reg_$0<llong x>)}}
  clang_analyzer_dump((char)(long)x);   // expected-warning{{(char) (reg_$0<llong x>)}}
  clang_analyzer_dump((char)(llong)x);  // expected-warning{{(char) (reg_$0<llong x>)}}
  clang_analyzer_dump((char)(uchar)x);  // expected-warning{{(char) (reg_$0<llong x>)}}
  clang_analyzer_dump((char)(ushort)x); // expected-warning{{(char) (reg_$0<llong x>)}}
  clang_analyzer_dump((char)(uint)x);   // expected-warning{{(char) (reg_$0<llong x>)}}
  clang_analyzer_dump((char)(ulong)x);  // expected-warning{{(char) (reg_$0<llong x>)}}
  clang_analyzer_dump((char)(ullong)x); // expected-warning{{(char) (reg_$0<llong x>)}}

  clang_analyzer_dump((short)(schar)x);  // expected-warning{{(short) ((signed char) (reg_$0<llong x>))}}
  clang_analyzer_dump((short)(char)x);   // expected-warning{{(short) ((char) (reg_$0<llong x>))}}
  clang_analyzer_dump((short)(short)x);  // expected-warning{{(short) (reg_$0<llong x>)}}
  clang_analyzer_dump((short)(int)x);    // expected-warning{{(short) (reg_$0<llong x>)}}
  clang_analyzer_dump((short)(long)x);   // expected-warning{{(short) (reg_$0<llong x>)}}
  clang_analyzer_dump((short)(llong)x);  // expected-warning{{(short) (reg_$0<llong x>)}}
  clang_analyzer_dump((short)(uchar)x);  // expected-warning{{(short) ((unsigned char) (reg_$0<llong x>))}}
  clang_analyzer_dump((short)(ushort)x); // expected-warning{{(short) (reg_$0<llong x>)}}
  clang_analyzer_dump((short)(uint)x);   // expected-warning{{(short) (reg_$0<llong x>)}}
  clang_analyzer_dump((short)(ulong)x);  // expected-warning{{(short) (reg_$0<llong x>)}}
  clang_analyzer_dump((short)(ullong)x); // expected-warning{{(short) (reg_$0<llong x>)}}

  clang_analyzer_dump((int)(schar)x);  // expected-warning{{(int) ((signed char) (reg_$0<llong x>))}}
  clang_analyzer_dump((int)(char)x);   // expected-warning{{(int) ((char) (reg_$0<llong x>))}}
  clang_analyzer_dump((int)(short)x);  // expected-warning{{(int) ((short) (reg_$0<llong x>))}}
  clang_analyzer_dump((int)(int)x);    // expected-warning{{(int) (reg_$0<llong x>)}}
  clang_analyzer_dump((int)(long)x);   // expected-warning{{(int) (reg_$0<llong x>)}}
  clang_analyzer_dump((int)(llong)x);  // expected-warning{{(int) (reg_$0<llong x>)}}
  clang_analyzer_dump((int)(uchar)x);  // expected-warning{{(int) ((unsigned char) (reg_$0<llong x>))}}
  clang_analyzer_dump((int)(ushort)x); // expected-warning{{(int) ((unsigned short) (reg_$0<llong x>))}}
  clang_analyzer_dump((int)(uint)x);   // expected-warning{{(int) (reg_$0<llong x>)}}
  clang_analyzer_dump((int)(ulong)x);  // expected-warning{{(int) (reg_$0<llong x>)}}
  clang_analyzer_dump((int)(ullong)x); // expected-warning{{(int) (reg_$0<llong x>)}}

  clang_analyzer_dump((long)(schar)x);  // expected-warning{{(long) ((signed char) (reg_$0<llong x>))}}
  clang_analyzer_dump((long)(char)x);   // expected-warning{{(long) ((char) (reg_$0<llong x>))}}
  clang_analyzer_dump((long)(short)x);  // expected-warning{{(long) ((short) (reg_$0<llong x>))}}
  clang_analyzer_dump((long)(int)x);    // expected-warning{{(long) ((int) (reg_$0<llong x>))}}
  clang_analyzer_dump((long)(long)x);   // expected-warning{{(long) (reg_$0<llong x>)}}
  clang_analyzer_dump((long)(llong)x);  // expected-warning{{(long) (reg_$0<llong x>)}}
  clang_analyzer_dump((long)(uchar)x);  // expected-warning{{(long) ((unsigned char) (reg_$0<llong x>))}}
  clang_analyzer_dump((long)(ushort)x); // expected-warning{{(long) ((unsigned short) (reg_$0<llong x>))}}
  clang_analyzer_dump((long)(uint)x);   // expected-warning{{(long) ((unsigned int) (reg_$0<llong x>))}}
  clang_analyzer_dump((long)(ulong)x);  // expected-warning{{(long) (reg_$0<llong x>)}}
  clang_analyzer_dump((long)(ullong)x); // expected-warning{{(long) (reg_$0<llong x>)}}

  clang_analyzer_dump((llong)(schar)x);  // expected-warning{{reg_$0<llong x>}}
  clang_analyzer_dump((llong)(char)x);   // expected-warning{{(char) (reg_$0<llong x>)}}
  clang_analyzer_dump((llong)(short)x);  // expected-warning{{(short) (reg_$0<llong x>)}}
  clang_analyzer_dump((llong)(int)x);    // expected-warning{{(int) (reg_$0<llong x>)}}
  clang_analyzer_dump((llong)(long)x);   // expected-warning{{reg_$0<llong x>}}
  clang_analyzer_dump((llong)(llong)x);  // expected-warning{{reg_$0<llong x>}}
  clang_analyzer_dump((llong)(uchar)x);  // expected-warning{{(unsigned char) (reg_$0<llong x>)}}
  clang_analyzer_dump((llong)(ushort)x); // expected-warning{{(unsigned short) (reg_$0<llong x>)}}
  clang_analyzer_dump((llong)(uint)x);   // expected-warning{{(unsigned int) (reg_$0<llong x>)}}
  clang_analyzer_dump((llong)(ulong)x);  // expected-warning{{reg_$0<llong x>}}
  clang_analyzer_dump((llong)(ullong)x); // expected-warning{{reg_$0<llong x>}}

  clang_analyzer_dump((uchar)(schar)x);  // expected-warning{{(unsigned char) (reg_$0<llong x>)}}
  clang_analyzer_dump((uchar)(char)x);   // expected-warning{{(unsigned char) (reg_$0<llong x>)}}
  clang_analyzer_dump((uchar)(short)x);  // expected-warning{{(unsigned char) (reg_$0<llong x>)}}
  clang_analyzer_dump((uchar)(int)x);    // expected-warning{{(unsigned char) (reg_$0<llong x>)}}
  clang_analyzer_dump((uchar)(long)x);   // expected-warning{{(unsigned char) (reg_$0<llong x>)}}
  clang_analyzer_dump((uchar)(llong)x);  // expected-warning{{(unsigned char) (reg_$0<llong x>)}}
  clang_analyzer_dump((uchar)(uchar)x);  // expected-warning{{(unsigned char) (reg_$0<llong x>)}}
  clang_analyzer_dump((uchar)(ushort)x); // expected-warning{{(unsigned char) (reg_$0<llong x>)}}
  clang_analyzer_dump((uchar)(uint)x);   // expected-warning{{(unsigned char) (reg_$0<llong x>)}}
  clang_analyzer_dump((uchar)(ulong)x);  // expected-warning{{(unsigned char) (reg_$0<llong x>)}}
  clang_analyzer_dump((uchar)(ullong)x); // expected-warning{{(unsigned char) (reg_$0<llong x>)}}

  clang_analyzer_dump((ushort)(schar)x);  // expected-warning{{(unsigned short) ((signed char) (reg_$0<llong x>))}}
  clang_analyzer_dump((ushort)(char)x);   // expected-warning{{(unsigned short) ((char) (reg_$0<llong x>))}}
  clang_analyzer_dump((ushort)(short)x);  // expected-warning{{(unsigned short) (reg_$0<llong x>)}}
  clang_analyzer_dump((ushort)(int)x);    // expected-warning{{(unsigned short) (reg_$0<llong x>)}}
  clang_analyzer_dump((ushort)(long)x);   // expected-warning{{(unsigned short) (reg_$0<llong x>)}}
  clang_analyzer_dump((ushort)(llong)x);  // expected-warning{{(unsigned short) (reg_$0<llong x>)}}
  clang_analyzer_dump((ushort)(uchar)x);  // expected-warning{{(unsigned short) ((unsigned char) (reg_$0<llong x>))}}
  clang_analyzer_dump((ushort)(ushort)x); // expected-warning{{(unsigned short) (reg_$0<llong x>)}}
  clang_analyzer_dump((ushort)(uint)x);   // expected-warning{{(unsigned short) (reg_$0<llong x>)}}
  clang_analyzer_dump((ushort)(ulong)x);  // expected-warning{{(unsigned short) (reg_$0<llong x>)}}
  clang_analyzer_dump((ushort)(ullong)x); // expected-warning{{(unsigned short) (reg_$0<llong x>)}}

  clang_analyzer_dump((uint)(schar)x);  // expected-warning{{(unsigned int) ((signed char) (reg_$0<llong x>))}}
  clang_analyzer_dump((uint)(char)x);   // expected-warning{{(unsigned int) ((char) (reg_$0<llong x>))}}
  clang_analyzer_dump((uint)(short)x);  // expected-warning{{(unsigned int) ((short) (reg_$0<llong x>))}}
  clang_analyzer_dump((uint)(int)x);    // expected-warning{{(unsigned int) (reg_$0<llong x>}}
  clang_analyzer_dump((uint)(long)x);   // expected-warning{{(unsigned int) (reg_$0<llong x>}}
  clang_analyzer_dump((uint)(llong)x);  // expected-warning{{(unsigned int) (reg_$0<llong x>}}
  clang_analyzer_dump((uint)(uchar)x);  // expected-warning{{(unsigned int) ((unsigned char) (reg_$0<llong x>))}}
  clang_analyzer_dump((uint)(ushort)x); // expected-warning{{(unsigned int) ((unsigned short) (reg_$0<llong x>))}}
  clang_analyzer_dump((uint)(uint)x);   // expected-warning{{(unsigned int) (reg_$0<llong x>}}
  clang_analyzer_dump((uint)(ulong)x);  // expected-warning{{(unsigned int) (reg_$0<llong x>}}
  clang_analyzer_dump((uint)(ullong)x); // expected-warning{{(unsigned int) (reg_$0<llong x>}}

  clang_analyzer_dump((ulong)(schar)x);  // expected-warning{{(unsigned long) ((signed char) (reg_$0<llong x>))}}
  clang_analyzer_dump((ulong)(char)x);   // expected-warning{{(unsigned long) ((char) (reg_$0<llong x>))}}
  clang_analyzer_dump((ulong)(short)x);  // expected-warning{{(unsigned long) ((short) (reg_$0<llong x>))}}
  clang_analyzer_dump((ulong)(int)x);    // expected-warning{{(unsigned long) ((int) (reg_$0<llong x>))}}
  clang_analyzer_dump((ulong)(long)x);   // expected-warning{{(unsigned long) (reg_$0<llong x>)}}
  clang_analyzer_dump((ulong)(llong)x);  // expected-warning{{(unsigned long) (reg_$0<llong x>)}}
  clang_analyzer_dump((ulong)(uchar)x);  // expected-warning{{(unsigned long) ((unsigned char) (reg_$0<llong x>))}}
  clang_analyzer_dump((ulong)(ushort)x); // expected-warning{{(unsigned long) ((unsigned short) (reg_$0<llong x>))}}
  clang_analyzer_dump((ulong)(uint)x);   // expected-warning{{(unsigned long) ((unsigned int) (reg_$0<llong x>))}}
  clang_analyzer_dump((ulong)(ulong)x);  // expected-warning{{(unsigned long) (reg_$0<llong x>)}}
  clang_analyzer_dump((ulong)(ullong)x); // expected-warning{{(unsigned long) (reg_$0<llong x>)}}

  clang_analyzer_dump((ullong)(schar)x);  // expected-warning{{(unsigned long long) ((signed char) (reg_$0<llong x>))}}
  clang_analyzer_dump((ullong)(char)x);   // expected-warning{{(unsigned long long) ((char) (reg_$0<llong x>))}}
  clang_analyzer_dump((ullong)(short)x);  // expected-warning{{(unsigned long long) ((short) (reg_$0<llong x>))}}
  clang_analyzer_dump((ullong)(int)x);    // expected-warning{{(unsigned long long) ((int) (reg_$0<llong x>))}}
  clang_analyzer_dump((ullong)(long)x);   // expected-warning{{(unsigned long long) (reg_$0<llong x>)}}
  clang_analyzer_dump((ullong)(llong)x);  // expected-warning{{(unsigned long long) (reg_$0<llong x>)}}
  clang_analyzer_dump((ullong)(uchar)x);  // expected-warning{{(unsigned long long) ((unsigned char) (reg_$0<llong x>))}}
  clang_analyzer_dump((ullong)(ushort)x); // expected-warning{{(unsigned long long) ((unsigned short) (reg_$0<llong x>))}}
  clang_analyzer_dump((ullong)(uint)x);   // expected-warning{{(unsigned long long) ((unsigned int) (reg_$0<llong x>))}}
  clang_analyzer_dump((ullong)(ulong)x);  // expected-warning{{(unsigned long long) (reg_$0<llong x>)}}
  clang_analyzer_dump((ullong)(ullong)x); // expected-warning{{(unsigned long long) (reg_$0<llong x>)}}
}

void test_uchar(uchar x) {
  clang_analyzer_dump(x); // expected-warning{{reg_$0<uchar x>}}

  clang_analyzer_dump((schar)x);  // expected-warning{{(signed char) (reg_$0<uchar x>)}}
  clang_analyzer_dump((char)x);   // expected-warning{{(char) (reg_$0<uchar x>)}}
  clang_analyzer_dump((short)x);  // expected-warning{{(short) (reg_$0<uchar x>)}}
  clang_analyzer_dump((int)x);    // expected-warning{{(int) (reg_$0<uchar x>)}}
  clang_analyzer_dump((long)x);   // expected-warning{{(long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((llong)x);  // expected-warning{{(long long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((uchar)x);  // expected-warning{{reg_$0<uchar x>}}
  clang_analyzer_dump((ushort)x); // expected-warning{{(unsigned short) (reg_$0<uchar x>)}}
  clang_analyzer_dump((uint)x);   // expected-warning{{(unsigned int) (reg_$0<uchar x>)}}
  clang_analyzer_dump((ulong)x);  // expected-warning{{(unsigned long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((ullong)x); // expected-warning{{(unsigned long long) (reg_$0<uchar x>)}}

  clang_analyzer_dump((schar)(schar)x);  // expected-warning{{(signed char) (reg_$0<uchar x>)}}
  clang_analyzer_dump((schar)(char)x);   // expected-warning{{(signed char) (reg_$0<uchar x>)}}
  clang_analyzer_dump((schar)(short)x);  // expected-warning{{(signed char) (reg_$0<uchar x>)}}
  clang_analyzer_dump((schar)(int)x);    // expected-warning{{(signed char) (reg_$0<uchar x>)}}
  clang_analyzer_dump((schar)(long)x);   // expected-warning{{(signed char) (reg_$0<uchar x>)}}
  clang_analyzer_dump((schar)(llong)x);  // expected-warning{{(signed char) (reg_$0<uchar x>)}}
  clang_analyzer_dump((schar)(uchar)x);  // expected-warning{{(signed char) (reg_$0<uchar x>)}}
  clang_analyzer_dump((schar)(ushort)x); // expected-warning{{(signed char) (reg_$0<uchar x>)}}
  clang_analyzer_dump((schar)(uint)x);   // expected-warning{{(signed char) (reg_$0<uchar x>)}}
  clang_analyzer_dump((schar)(ulong)x);  // expected-warning{{(signed char) (reg_$0<uchar x>)}}
  clang_analyzer_dump((schar)(ullong)x); // expected-warning{{(signed char) (reg_$0<uchar x>)}}

  clang_analyzer_dump((char)(schar)x);  // expected-warning{{(char) (reg_$0<uchar x>)}}
  clang_analyzer_dump((char)(char)x);   // expected-warning{{(char) (reg_$0<uchar x>)}}
  clang_analyzer_dump((char)(short)x);  // expected-warning{{(char) (reg_$0<uchar x>)}}
  clang_analyzer_dump((char)(int)x);    // expected-warning{{(char) (reg_$0<uchar x>)}}
  clang_analyzer_dump((char)(long)x);   // expected-warning{{(char) (reg_$0<uchar x>)}}
  clang_analyzer_dump((char)(llong)x);  // expected-warning{{(char) (reg_$0<uchar x>)}}
  clang_analyzer_dump((char)(uchar)x);  // expected-warning{{(char) (reg_$0<uchar x>)}}
  clang_analyzer_dump((char)(ushort)x); // expected-warning{{(char) (reg_$0<uchar x>)}}
  clang_analyzer_dump((char)(uint)x);   // expected-warning{{(char) (reg_$0<uchar x>)}}
  clang_analyzer_dump((char)(ulong)x);  // expected-warning{{(char) (reg_$0<uchar x>)}}
  clang_analyzer_dump((char)(ullong)x); // expected-warning{{(char) (reg_$0<uchar x>)}}

  clang_analyzer_dump((short)(schar)x);  // expected-warning{{(short) ((signed char) (reg_$0<uchar x>))}}
  clang_analyzer_dump((short)(char)x);   // expected-warning{{(short) ((char) (reg_$0<uchar x>))}}
  clang_analyzer_dump((short)(short)x);  // expected-warning{{(short) (reg_$0<uchar x>)}}
  clang_analyzer_dump((short)(int)x);    // expected-warning{{(short) (reg_$0<uchar x>)}}
  clang_analyzer_dump((short)(long)x);   // expected-warning{{(short) (reg_$0<uchar x>)}}
  clang_analyzer_dump((short)(llong)x);  // expected-warning{{(short) (reg_$0<uchar x>)}}
  clang_analyzer_dump((short)(uchar)x);  // expected-warning{{(short) (reg_$0<uchar x>)}}
  clang_analyzer_dump((short)(ushort)x); // expected-warning{{(short) (reg_$0<uchar x>)}}
  clang_analyzer_dump((short)(uint)x);   // expected-warning{{(short) (reg_$0<uchar x>)}}
  clang_analyzer_dump((short)(ulong)x);  // expected-warning{{(short) (reg_$0<uchar x>)}}
  clang_analyzer_dump((short)(ullong)x); // expected-warning{{(short) (reg_$0<uchar x>)}}

  clang_analyzer_dump((int)(schar)x);  // expected-warning{{(int) ((signed char) (reg_$0<uchar x>))}}
  clang_analyzer_dump((int)(char)x);   // expected-warning{{(int) ((char) (reg_$0<uchar x>))}}
  clang_analyzer_dump((int)(short)x);  // expected-warning{{(int) (reg_$0<uchar x>)}}
  clang_analyzer_dump((int)(int)x);    // expected-warning{{(int) (reg_$0<uchar x>)}}
  clang_analyzer_dump((int)(long)x);   // expected-warning{{(int) (reg_$0<uchar x>)}}
  clang_analyzer_dump((int)(llong)x);  // expected-warning{{(int) (reg_$0<uchar x>)}}
  clang_analyzer_dump((int)(uchar)x);  // expected-warning{{(int) (reg_$0<uchar x>)}}
  clang_analyzer_dump((int)(ushort)x); // expected-warning{{(int) (reg_$0<uchar x>)}}
  clang_analyzer_dump((int)(uint)x);   // expected-warning{{(int) (reg_$0<uchar x>)}}
  clang_analyzer_dump((int)(ulong)x);  // expected-warning{{(int) (reg_$0<uchar x>)}}
  clang_analyzer_dump((int)(ullong)x); // expected-warning{{(int) (reg_$0<uchar x>)}}

  clang_analyzer_dump((long)(schar)x);  // expected-warning{{(long) ((signed char) (reg_$0<uchar x>))}}
  clang_analyzer_dump((long)(char)x);   // expected-warning{{(long) ((char) (reg_$0<uchar x>))}}
  clang_analyzer_dump((long)(short)x);  // expected-warning{{(long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((long)(int)x);    // expected-warning{{(long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((long)(long)x);   // expected-warning{{(long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((long)(llong)x);  // expected-warning{{(long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((long)(uchar)x);  // expected-warning{{(long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((long)(ushort)x); // expected-warning{{(long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((long)(uint)x);   // expected-warning{{(long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((long)(ulong)x);  // expected-warning{{(long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((long)(ullong)x); // expected-warning{{(long) (reg_$0<uchar x>)}}

  clang_analyzer_dump((llong)(schar)x);  // expected-warning{{(long long) ((signed char) (reg_$0<uchar x>))}}
  clang_analyzer_dump((llong)(char)x);   // expected-warning{{(long long) ((char) (reg_$0<uchar x>))}}
  clang_analyzer_dump((llong)(short)x);  // expected-warning{{(long long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((llong)(int)x);    // expected-warning{{(long long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((llong)(long)x);   // expected-warning{{(long long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((llong)(llong)x);  // expected-warning{{(long long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((llong)(uchar)x);  // expected-warning{{(long long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((llong)(ushort)x); // expected-warning{{(long long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((llong)(uint)x);   // expected-warning{{(long long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((llong)(ulong)x);  // expected-warning{{(long long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((llong)(ullong)x); // expected-warning{{(long long) (reg_$0<uchar x>)}}

  clang_analyzer_dump((uchar)(schar)x);  // expected-warning{{reg_$0<uchar x>}}
  clang_analyzer_dump((uchar)(char)x);   // expected-warning{{reg_$0<uchar x>}}
  clang_analyzer_dump((uchar)(short)x);  // expected-warning{{reg_$0<uchar x>}}
  clang_analyzer_dump((uchar)(int)x);    // expected-warning{{reg_$0<uchar x>}}
  clang_analyzer_dump((uchar)(long)x);   // expected-warning{{reg_$0<uchar x>}}
  clang_analyzer_dump((uchar)(llong)x);  // expected-warning{{reg_$0<uchar x>}}
  clang_analyzer_dump((uchar)(uchar)x);  // expected-warning{{reg_$0<uchar x>}}
  clang_analyzer_dump((uchar)(ushort)x); // expected-warning{{reg_$0<uchar x>}}
  clang_analyzer_dump((uchar)(uint)x);   // expected-warning{{reg_$0<uchar x>}}
  clang_analyzer_dump((uchar)(ulong)x);  // expected-warning{{reg_$0<uchar x>}}
  clang_analyzer_dump((uchar)(ullong)x); // expected-warning{{reg_$0<uchar x>}}

  clang_analyzer_dump((ushort)(schar)x);  // expected-warning{{(unsigned short) ((signed char) (reg_$0<uchar x>))}}
  clang_analyzer_dump((ushort)(char)x);   // expected-warning{{(unsigned short) ((char) (reg_$0<uchar x>))}}
  clang_analyzer_dump((ushort)(short)x);  // expected-warning{{(unsigned short) (reg_$0<uchar x>)}}
  clang_analyzer_dump((ushort)(int)x);    // expected-warning{{(unsigned short) (reg_$0<uchar x>)}}
  clang_analyzer_dump((ushort)(long)x);   // expected-warning{{(unsigned short) (reg_$0<uchar x>)}}
  clang_analyzer_dump((ushort)(llong)x);  // expected-warning{{(unsigned short) (reg_$0<uchar x>)}}
  clang_analyzer_dump((ushort)(uchar)x);  // expected-warning{{(unsigned short) (reg_$0<uchar x>)}}
  clang_analyzer_dump((ushort)(ushort)x); // expected-warning{{(unsigned short) (reg_$0<uchar x>)}}
  clang_analyzer_dump((ushort)(uint)x);   // expected-warning{{(unsigned short) (reg_$0<uchar x>)}}
  clang_analyzer_dump((ushort)(ulong)x);  // expected-warning{{(unsigned short) (reg_$0<uchar x>)}}
  clang_analyzer_dump((ushort)(ullong)x); // expected-warning{{(unsigned short) (reg_$0<uchar x>)}}

  clang_analyzer_dump((uint)(schar)x);  // expected-warning{{(unsigned int) ((signed char) (reg_$0<uchar x>))}}
  clang_analyzer_dump((uint)(char)x);   // expected-warning{{(unsigned int) ((char) (reg_$0<uchar x>))}}
  clang_analyzer_dump((uint)(short)x);  // expected-warning{{(unsigned int) (reg_$0<uchar x>)}}
  clang_analyzer_dump((uint)(int)x);    // expected-warning{{(unsigned int) (reg_$0<uchar x>)}}
  clang_analyzer_dump((uint)(long)x);   // expected-warning{{(unsigned int) (reg_$0<uchar x>)}}
  clang_analyzer_dump((uint)(llong)x);  // expected-warning{{(unsigned int) (reg_$0<uchar x>)}}
  clang_analyzer_dump((uint)(uchar)x);  // expected-warning{{(unsigned int) (reg_$0<uchar x>)}}
  clang_analyzer_dump((uint)(ushort)x); // expected-warning{{(unsigned int) (reg_$0<uchar x>)}}
  clang_analyzer_dump((uint)(uint)x);   // expected-warning{{(unsigned int) (reg_$0<uchar x>)}}
  clang_analyzer_dump((uint)(ulong)x);  // expected-warning{{(unsigned int) (reg_$0<uchar x>)}}
  clang_analyzer_dump((uint)(ullong)x); // expected-warning{{(unsigned int) (reg_$0<uchar x>)}}

  clang_analyzer_dump((ulong)(schar)x);  // expected-warning{{(unsigned long) ((signed char) (reg_$0<uchar x>))}}
  clang_analyzer_dump((ulong)(char)x);   // expected-warning{{(unsigned long) ((char) (reg_$0<uchar x>))}}
  clang_analyzer_dump((ulong)(short)x);  // expected-warning{{(unsigned long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((ulong)(int)x);    // expected-warning{{(unsigned long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((ulong)(long)x);   // expected-warning{{(unsigned long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((ulong)(llong)x);  // expected-warning{{(unsigned long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((ulong)(uchar)x);  // expected-warning{{(unsigned long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((ulong)(ushort)x); // expected-warning{{(unsigned long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((ulong)(uint)x);   // expected-warning{{(unsigned long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((ulong)(ulong)x);  // expected-warning{{(unsigned long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((ulong)(ullong)x); // expected-warning{{(unsigned long) (reg_$0<uchar x>)}}

  clang_analyzer_dump((ullong)(schar)x);  // expected-warning{{(unsigned long long) ((signed char) (reg_$0<uchar x>))}}
  clang_analyzer_dump((ullong)(char)x);   // expected-warning{{(unsigned long long) ((char) (reg_$0<uchar x>))}}
  clang_analyzer_dump((ullong)(short)x);  // expected-warning{{(unsigned long long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((ullong)(int)x);    // expected-warning{{(unsigned long long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((ullong)(long)x);   // expected-warning{{(unsigned long long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((ullong)(llong)x);  // expected-warning{{(unsigned long long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((ullong)(uchar)x);  // expected-warning{{(unsigned long long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((ullong)(ushort)x); // expected-warning{{(unsigned long long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((ullong)(uint)x);   // expected-warning{{(unsigned long long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((ullong)(ulong)x);  // expected-warning{{(unsigned long long) (reg_$0<uchar x>)}}
  clang_analyzer_dump((ullong)(ullong)x); // expected-warning{{(unsigned long long) (reg_$0<uchar x>)}}
}

void test_ushort(ushort x) {
  clang_analyzer_dump(x); // expected-warning{{reg_$0<ushort x>}}

  clang_analyzer_dump((schar)x);  // expected-warning{{(signed char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((char)x);   // expected-warning{{(char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((short)x);  // expected-warning{{(short) (reg_$0<ushort x>)}}
  clang_analyzer_dump((int)x);    // expected-warning{{(int) (reg_$0<ushort x>)}}
  clang_analyzer_dump((long)x);   // expected-warning{{(long) (reg_$0<ushort x>)}}
  clang_analyzer_dump((llong)x);  // expected-warning{{(long long) (reg_$0<ushort x>)}}
  clang_analyzer_dump((uchar)x);  // expected-warning{{(unsigned char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((ushort)x); // expected-warning{{reg_$0<ushort x>}}
  clang_analyzer_dump((uint)x);   // expected-warning{{(unsigned int) (reg_$0<ushort x>)}}
  clang_analyzer_dump((ulong)x);  // expected-warning{{(unsigned long) (reg_$0<ushort x>)}}
  clang_analyzer_dump((ullong)x); // expected-warning{{(unsigned long long) (reg_$0<ushort x>)}}

  clang_analyzer_dump((schar)(schar)x);  // expected-warning{{(signed char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((schar)(char)x);   // expected-warning{{(signed char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((schar)(short)x);  // expected-warning{{(signed char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((schar)(int)x);    // expected-warning{{(signed char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((schar)(long)x);   // expected-warning{{(signed char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((schar)(llong)x);  // expected-warning{{(signed char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((schar)(uchar)x);  // expected-warning{{(signed char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((schar)(ushort)x); // expected-warning{{(signed char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((schar)(uint)x);   // expected-warning{{(signed char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((schar)(ulong)x);  // expected-warning{{(signed char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((schar)(ullong)x); // expected-warning{{(signed char) (reg_$0<ushort x>)}}

  clang_analyzer_dump((char)(schar)x);  // expected-warning{{(char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((char)(char)x);   // expected-warning{{(char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((char)(short)x);  // expected-warning{{(char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((char)(int)x);    // expected-warning{{(char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((char)(long)x);   // expected-warning{{(char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((char)(llong)x);  // expected-warning{{(char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((char)(uchar)x);  // expected-warning{{(char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((char)(ushort)x); // expected-warning{{(char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((char)(uint)x);   // expected-warning{{(char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((char)(ulong)x);  // expected-warning{{(char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((char)(ullong)x); // expected-warning{{(char) (reg_$0<ushort x>)}}

  clang_analyzer_dump((short)(schar)x);  // expected-warning{{(short) ((signed char) (reg_$0<ushort x>))}}
  clang_analyzer_dump((short)(char)x);   // expected-warning{{(short) ((char) (reg_$0<ushort x>))}}
  clang_analyzer_dump((short)(short)x);  // expected-warning{{(short) (reg_$0<ushort x>)}}
  clang_analyzer_dump((short)(int)x);    // expected-warning{{(short) (reg_$0<ushort x>)}}
  clang_analyzer_dump((short)(long)x);   // expected-warning{{(short) (reg_$0<ushort x>)}}
  clang_analyzer_dump((short)(llong)x);  // expected-warning{{(short) (reg_$0<ushort x>)}}
  clang_analyzer_dump((short)(uchar)x);  // expected-warning{{(short) ((unsigned char) (reg_$0<ushort x>))}}
  clang_analyzer_dump((short)(ushort)x); // expected-warning{{(short) (reg_$0<ushort x>)}}
  clang_analyzer_dump((short)(uint)x);   // expected-warning{{(short) (reg_$0<ushort x>)}}
  clang_analyzer_dump((short)(ulong)x);  // expected-warning{{(short) (reg_$0<ushort x>)}}
  clang_analyzer_dump((short)(ullong)x); // expected-warning{{(short) (reg_$0<ushort x>)}}

  clang_analyzer_dump((int)(schar)x);  // expected-warning{{(int) ((signed char) (reg_$0<ushort x>))}}
  clang_analyzer_dump((int)(char)x);   // expected-warning{{(int) ((char) (reg_$0<ushort x>))}}
  clang_analyzer_dump((int)(short)x);  // expected-warning{{(int) ((short) (reg_$0<ushort x>))}}
  clang_analyzer_dump((int)(int)x);    // expected-warning{{(int) (reg_$0<ushort x>)}}
  clang_analyzer_dump((int)(long)x);   // expected-warning{{(int) (reg_$0<ushort x>)}}
  clang_analyzer_dump((int)(llong)x);  // expected-warning{{(int) (reg_$0<ushort x>)}}
  clang_analyzer_dump((int)(uchar)x);  // expected-warning{{(int) ((unsigned char) (reg_$0<ushort x>))}}
  clang_analyzer_dump((int)(ushort)x); // expected-warning{{(int) (reg_$0<ushort x>)}}
  clang_analyzer_dump((int)(uint)x);   // expected-warning{{(int) (reg_$0<ushort x>)}}
  clang_analyzer_dump((int)(ulong)x);  // expected-warning{{(int) (reg_$0<ushort x>)}}
  clang_analyzer_dump((int)(ullong)x); // expected-warning{{(int) (reg_$0<ushort x>)}}

  clang_analyzer_dump((long)(schar)x);  // expected-warning{{(long) ((signed char) (reg_$0<ushort x>))}}
  clang_analyzer_dump((long)(char)x);   // expected-warning{{(long) ((char) (reg_$0<ushort x>))}}
  clang_analyzer_dump((long)(short)x);  // expected-warning{{(long) ((short) (reg_$0<ushort x>))}}
  clang_analyzer_dump((long)(int)x);    // expected-warning{{(long) (reg_$0<ushort x>)}}
  clang_analyzer_dump((long)(long)x);   // expected-warning{{(long) (reg_$0<ushort x>)}}
  clang_analyzer_dump((long)(llong)x);  // expected-warning{{(long) (reg_$0<ushort x>)}}
  clang_analyzer_dump((long)(uchar)x);  // expected-warning{{(long) ((unsigned char) (reg_$0<ushort x>))}}
  clang_analyzer_dump((long)(ushort)x); // expected-warning{{(long) (reg_$0<ushort x>)}}
  clang_analyzer_dump((long)(uint)x);   // expected-warning{{(long) (reg_$0<ushort x>)}}
  clang_analyzer_dump((long)(ulong)x);  // expected-warning{{(long) (reg_$0<ushort x>)}}
  clang_analyzer_dump((long)(ullong)x); // expected-warning{{(long) (reg_$0<ushort x>)}}

  clang_analyzer_dump((llong)(schar)x);  // expected-warning{{(long long) ((signed char) (reg_$0<ushort x>))}}
  clang_analyzer_dump((llong)(char)x);   // expected-warning{{(long long) ((char) (reg_$0<ushort x>))}}
  clang_analyzer_dump((llong)(short)x);  // expected-warning{{(long long) ((short) (reg_$0<ushort x>))}}
  clang_analyzer_dump((llong)(int)x);    // expected-warning{{(long long) (reg_$0<ushort x>)}}
  clang_analyzer_dump((llong)(long)x);   // expected-warning{{(long long) (reg_$0<ushort x>)}}
  clang_analyzer_dump((llong)(llong)x);  // expected-warning{{(long long) (reg_$0<ushort x>)}}
  clang_analyzer_dump((llong)(uchar)x);  // expected-warning{{(long long) ((unsigned char) (reg_$0<ushort x>))}}
  clang_analyzer_dump((llong)(ushort)x); // expected-warning{{(long long) (reg_$0<ushort x>)}}
  clang_analyzer_dump((llong)(uint)x);   // expected-warning{{(long long) (reg_$0<ushort x>)}}
  clang_analyzer_dump((llong)(ulong)x);  // expected-warning{{(long long) (reg_$0<ushort x>)}}
  clang_analyzer_dump((llong)(ullong)x); // expected-warning{{(long long) (reg_$0<ushort x>)}}

  clang_analyzer_dump((uchar)(schar)x);  // expected-warning{{(unsigned char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((uchar)(char)x);   // expected-warning{{(unsigned char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((uchar)(short)x);  // expected-warning{{(unsigned char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((uchar)(int)x);    // expected-warning{{(unsigned char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((uchar)(long)x);   // expected-warning{{(unsigned char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((uchar)(llong)x);  // expected-warning{{(unsigned char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((uchar)(uchar)x);  // expected-warning{{(unsigned char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((uchar)(ushort)x); // expected-warning{{(unsigned char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((uchar)(uint)x);   // expected-warning{{(unsigned char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((uchar)(ulong)x);  // expected-warning{{(unsigned char) (reg_$0<ushort x>)}}
  clang_analyzer_dump((uchar)(ullong)x); // expected-warning{{(unsigned char) (reg_$0<ushort x>)}}

  clang_analyzer_dump((ushort)(schar)x);  // expected-warning{{(unsigned short) ((signed char) (reg_$0<ushort x>))}}
  clang_analyzer_dump((ushort)(char)x);   // expected-warning{{(unsigned short) ((char) (reg_$0<ushort x>))}}
  clang_analyzer_dump((ushort)(short)x);  // expected-warning{{reg_$0<ushort x>}}
  clang_analyzer_dump((ushort)(int)x);    // expected-warning{{reg_$0<ushort x>}}
  clang_analyzer_dump((ushort)(long)x);   // expected-warning{{reg_$0<ushort x>}}
  clang_analyzer_dump((ushort)(llong)x);  // expected-warning{{reg_$0<ushort x>}}
  clang_analyzer_dump((ushort)(uchar)x);  // expected-warning{{(unsigned short) ((unsigned char) (reg_$0<ushort x>))}}
  clang_analyzer_dump((ushort)(ushort)x); // expected-warning{{reg_$0<ushort x>}}
  clang_analyzer_dump((ushort)(uint)x);   // expected-warning{{reg_$0<ushort x>}}
  clang_analyzer_dump((ushort)(ulong)x);  // expected-warning{{reg_$0<ushort x>}}
  clang_analyzer_dump((ushort)(ullong)x); // expected-warning{{reg_$0<ushort x>}}

  clang_analyzer_dump((uint)(schar)x);  // expected-warning{{(unsigned int) ((signed char) (reg_$0<ushort x>))}}
  clang_analyzer_dump((uint)(char)x);   // expected-warning{{(unsigned int) ((char) (reg_$0<ushort x>))}}
  clang_analyzer_dump((uint)(short)x);  // expected-warning{{(unsigned int) ((short) (reg_$0<ushort x>))}}
  clang_analyzer_dump((uint)(int)x);    // expected-warning{{(unsigned int) (reg_$0<ushort x>)}}
  clang_analyzer_dump((uint)(long)x);   // expected-warning{{(unsigned int) (reg_$0<ushort x>)}}
  clang_analyzer_dump((uint)(llong)x);  // expected-warning{{(unsigned int) (reg_$0<ushort x>)}}
  clang_analyzer_dump((uint)(uchar)x);  // expected-warning{{(unsigned int) ((unsigned char) (reg_$0<ushort x>))}}
  clang_analyzer_dump((uint)(ushort)x); // expected-warning{{(unsigned int) (reg_$0<ushort x>)}}
  clang_analyzer_dump((uint)(uint)x);   // expected-warning{{(unsigned int) (reg_$0<ushort x>)}}
  clang_analyzer_dump((uint)(ulong)x);  // expected-warning{{(unsigned int) (reg_$0<ushort x>)}}
  clang_analyzer_dump((uint)(ullong)x); // expected-warning{{(unsigned int) (reg_$0<ushort x>)}}

  clang_analyzer_dump((ulong)(schar)x);  // expected-warning{{(unsigned long) ((signed char) (reg_$0<ushort x>))}}
  clang_analyzer_dump((ulong)(char)x);   // expected-warning{{(unsigned long) ((char) (reg_$0<ushort x>))}}
  clang_analyzer_dump((ulong)(short)x);  // expected-warning{{(unsigned long) ((short) (reg_$0<ushort x>))}}
  clang_analyzer_dump((ulong)(int)x);    // expected-warning{{(unsigned long) (reg_$0<ushort x>)}}
  clang_analyzer_dump((ulong)(long)x);   // expected-warning{{(unsigned long) (reg_$0<ushort x>)}}
  clang_analyzer_dump((ulong)(llong)x);  // expected-warning{{(unsigned long) (reg_$0<ushort x>)}}
  clang_analyzer_dump((ulong)(uchar)x);  // expected-warning{{(unsigned long) ((unsigned char) (reg_$0<ushort x>))}}
  clang_analyzer_dump((ulong)(ushort)x); // expected-warning{{(unsigned long) (reg_$0<ushort x>)}}
  clang_analyzer_dump((ulong)(uint)x);   // expected-warning{{(unsigned long) (reg_$0<ushort x>)}}
  clang_analyzer_dump((ulong)(ulong)x);  // expected-warning{{(unsigned long) (reg_$0<ushort x>)}}
  clang_analyzer_dump((ulong)(ullong)x); // expected-warning{{(unsigned long) (reg_$0<ushort x>)}}

  clang_analyzer_dump((ullong)(schar)x);  // expected-warning{{(unsigned long long) ((signed char) (reg_$0<ushort x>))}}
  clang_analyzer_dump((ullong)(char)x);   // expected-warning{{(unsigned long long) ((char) (reg_$0<ushort x>))}}
  clang_analyzer_dump((ullong)(short)x);  // expected-warning{{(unsigned long long) ((short) (reg_$0<ushort x>))}}
  clang_analyzer_dump((ullong)(int)x);    // expected-warning{{(unsigned long long) (reg_$0<ushort x>)}}
  clang_analyzer_dump((ullong)(long)x);   // expected-warning{{(unsigned long long) (reg_$0<ushort x>)}}
  clang_analyzer_dump((ullong)(llong)x);  // expected-warning{{(unsigned long long) (reg_$0<ushort x>)}}
  clang_analyzer_dump((ullong)(uchar)x);  // expected-warning{{(unsigned long long) ((unsigned char) (reg_$0<ushort x>))}}
  clang_analyzer_dump((ullong)(ushort)x); // expected-warning{{(unsigned long long) (reg_$0<ushort x>)}}
  clang_analyzer_dump((ullong)(uint)x);   // expected-warning{{(unsigned long long) (reg_$0<ushort x>)}}
  clang_analyzer_dump((ullong)(ulong)x);  // expected-warning{{(unsigned long long) (reg_$0<ushort x>)}}
  clang_analyzer_dump((ullong)(ullong)x); // expected-warning{{(unsigned long long) (reg_$0<ushort x>)}}
}

void test_uint(uint x) {
  clang_analyzer_dump(x); // expected-warning{{reg_$0<uint x>}}

  clang_analyzer_dump((schar)x);  // expected-warning{{(signed char) (reg_$0<uint x>)}}
  clang_analyzer_dump((char)x);   // expected-warning{{(char) (reg_$0<uint x>)}}
  clang_analyzer_dump((short)x);  // expected-warning{{(short) (reg_$0<uint x>)}}
  clang_analyzer_dump((int)x);    // expected-warning{{(int) (reg_$0<uint x>)}}
  clang_analyzer_dump((long)x);   // expected-warning{{(long) (reg_$0<uint x>)}}
  clang_analyzer_dump((llong)x);  // expected-warning{{(long long) (reg_$0<uint x>)}}
  clang_analyzer_dump((uchar)x);  // expected-warning{{(unsigned char) (reg_$0<uint x>)}}
  clang_analyzer_dump((ushort)x); // expected-warning{{(unsigned short) (reg_$0<uint x>)}}
  clang_analyzer_dump((uint)x);   // expected-warning{{reg_$0<uint x>}}
  clang_analyzer_dump((ulong)x);  // expected-warning{{(unsigned long) (reg_$0<uint x>)}}
  clang_analyzer_dump((ullong)x); // expected-warning{{(unsigned long long) (reg_$0<uint x>)}}

  clang_analyzer_dump((schar)(schar)x);  // expected-warning{{(signed char) (reg_$0<uint x>)}}
  clang_analyzer_dump((schar)(char)x);   // expected-warning{{(signed char) (reg_$0<uint x>)}}
  clang_analyzer_dump((schar)(short)x);  // expected-warning{{(signed char) (reg_$0<uint x>)}}
  clang_analyzer_dump((schar)(int)x);    // expected-warning{{(signed char) (reg_$0<uint x>)}}
  clang_analyzer_dump((schar)(long)x);   // expected-warning{{(signed char) (reg_$0<uint x>)}}
  clang_analyzer_dump((schar)(llong)x);  // expected-warning{{(signed char) (reg_$0<uint x>)}}
  clang_analyzer_dump((schar)(uchar)x);  // expected-warning{{(signed char) (reg_$0<uint x>)}}
  clang_analyzer_dump((schar)(ushort)x); // expected-warning{{(signed char) (reg_$0<uint x>)}}
  clang_analyzer_dump((schar)(uint)x);   // expected-warning{{(signed char) (reg_$0<uint x>)}}
  clang_analyzer_dump((schar)(ulong)x);  // expected-warning{{(signed char) (reg_$0<uint x>)}}
  clang_analyzer_dump((schar)(ullong)x); // expected-warning{{(signed char) (reg_$0<uint x>)}}

  clang_analyzer_dump((char)(schar)x);  // expected-warning{{(char) (reg_$0<uint x>)}}
  clang_analyzer_dump((char)(char)x);   // expected-warning{{(char) (reg_$0<uint x>)}}
  clang_analyzer_dump((char)(short)x);  // expected-warning{{(char) (reg_$0<uint x>)}}
  clang_analyzer_dump((char)(int)x);    // expected-warning{{(char) (reg_$0<uint x>)}}
  clang_analyzer_dump((char)(long)x);   // expected-warning{{(char) (reg_$0<uint x>)}}
  clang_analyzer_dump((char)(llong)x);  // expected-warning{{(char) (reg_$0<uint x>)}}
  clang_analyzer_dump((char)(uchar)x);  // expected-warning{{(char) (reg_$0<uint x>)}}
  clang_analyzer_dump((char)(ushort)x); // expected-warning{{(char) (reg_$0<uint x>)}}
  clang_analyzer_dump((char)(uint)x);   // expected-warning{{(char) (reg_$0<uint x>)}}
  clang_analyzer_dump((char)(ulong)x);  // expected-warning{{(char) (reg_$0<uint x>)}}
  clang_analyzer_dump((char)(ullong)x); // expected-warning{{(char) (reg_$0<uint x>)}}

  clang_analyzer_dump((short)(schar)x);  // expected-warning{{(short) ((signed char) (reg_$0<uint x>))}}
  clang_analyzer_dump((short)(char)x);   // expected-warning{{(short) ((char) (reg_$0<uint x>))}}
  clang_analyzer_dump((short)(short)x);  // expected-warning{{(short) (reg_$0<uint x>)}}
  clang_analyzer_dump((short)(int)x);    // expected-warning{{(short) (reg_$0<uint x>)}}
  clang_analyzer_dump((short)(long)x);   // expected-warning{{(short) (reg_$0<uint x>)}}
  clang_analyzer_dump((short)(llong)x);  // expected-warning{{(short) (reg_$0<uint x>)}}
  clang_analyzer_dump((short)(uchar)x);  // expected-warning{{(short) ((unsigned char) (reg_$0<uint x>))}}
  clang_analyzer_dump((short)(ushort)x); // expected-warning{{(short) (reg_$0<uint x>)}}
  clang_analyzer_dump((short)(uint)x);   // expected-warning{{(short) (reg_$0<uint x>)}}
  clang_analyzer_dump((short)(ulong)x);  // expected-warning{{(short) (reg_$0<uint x>)}}
  clang_analyzer_dump((short)(ullong)x); // expected-warning{{(short) (reg_$0<uint x>)}}

  clang_analyzer_dump((int)(schar)x);  // expected-warning{{(int) ((signed char) (reg_$0<uint x>))}}
  clang_analyzer_dump((int)(char)x);   // expected-warning{{(int) ((char) (reg_$0<uint x>))}}
  clang_analyzer_dump((int)(short)x);  // expected-warning{{(int) ((short) (reg_$0<uint x>))}}
  clang_analyzer_dump((int)(int)x);    // expected-warning{{(int) (reg_$0<uint x>)}}
  clang_analyzer_dump((int)(long)x);   // expected-warning{{(int) (reg_$0<uint x>)}}
  clang_analyzer_dump((int)(llong)x);  // expected-warning{{(int) (reg_$0<uint x>)}}
  clang_analyzer_dump((int)(uchar)x);  // expected-warning{{(int) ((unsigned char) (reg_$0<uint x>))}}
  clang_analyzer_dump((int)(ushort)x); // expected-warning{{(int) ((unsigned short) (reg_$0<uint x>))}}
  clang_analyzer_dump((int)(uint)x);   // expected-warning{{(int) (reg_$0<uint x>)}}
  clang_analyzer_dump((int)(ulong)x);  // expected-warning{{(int) (reg_$0<uint x>)}}
  clang_analyzer_dump((int)(ullong)x); // expected-warning{{(int) (reg_$0<uint x>)}}

  clang_analyzer_dump((long)(schar)x);  // expected-warning{{(long) ((signed char) (reg_$0<uint x>))}}
  clang_analyzer_dump((long)(char)x);   // expected-warning{{(long) ((char) (reg_$0<uint x>))}}
  clang_analyzer_dump((long)(short)x);  // expected-warning{{(long) ((short) (reg_$0<uint x>))}}
  clang_analyzer_dump((long)(int)x);    // expected-warning{{(long) ((int) (reg_$0<uint x>))}}
  clang_analyzer_dump((long)(long)x);   // expected-warning{{(long) (reg_$0<uint x>)}}
  clang_analyzer_dump((long)(llong)x);  // expected-warning{{(long) (reg_$0<uint x>)}}
  clang_analyzer_dump((long)(uchar)x);  // expected-warning{{(long) ((unsigned char) (reg_$0<uint x>))}}
  clang_analyzer_dump((long)(ushort)x); // expected-warning{{(long) ((unsigned short) (reg_$0<uint x>))}}
  clang_analyzer_dump((long)(uint)x);   // expected-warning{{(long) (reg_$0<uint x>)}}
  clang_analyzer_dump((long)(ulong)x);  // expected-warning{{(long) (reg_$0<uint x>)}}
  clang_analyzer_dump((long)(ullong)x); // expected-warning{{(long) (reg_$0<uint x>)}}

  clang_analyzer_dump((llong)(schar)x);  // expected-warning{{(long long) ((signed char) (reg_$0<uint x>))}}
  clang_analyzer_dump((llong)(char)x);   // expected-warning{{(long long) ((char) (reg_$0<uint x>))}}
  clang_analyzer_dump((llong)(short)x);  // expected-warning{{(long long) ((short) (reg_$0<uint x>))}}
  clang_analyzer_dump((llong)(int)x);    // expected-warning{{(long long) ((int) (reg_$0<uint x>))}}
  clang_analyzer_dump((llong)(long)x);   // expected-warning{{(long long) (reg_$0<uint x>)}}
  clang_analyzer_dump((llong)(llong)x);  // expected-warning{{(long long) (reg_$0<uint x>)}}
  clang_analyzer_dump((llong)(uchar)x);  // expected-warning{{(long long) ((unsigned char) (reg_$0<uint x>))}}
  clang_analyzer_dump((llong)(ushort)x); // expected-warning{{(long long) ((unsigned short) (reg_$0<uint x>))}}
  clang_analyzer_dump((llong)(uint)x);   // expected-warning{{(long long) (reg_$0<uint x>)}}
  clang_analyzer_dump((llong)(ulong)x);  // expected-warning{{(long long) (reg_$0<uint x>)}}
  clang_analyzer_dump((llong)(ullong)x); // expected-warning{{(long long) (reg_$0<uint x>)}}

  clang_analyzer_dump((uchar)(schar)x);  // expected-warning{{(unsigned char) (reg_$0<uint x>)}}
  clang_analyzer_dump((uchar)(char)x);   // expected-warning{{(unsigned char) (reg_$0<uint x>)}}
  clang_analyzer_dump((uchar)(short)x);  // expected-warning{{(unsigned char) (reg_$0<uint x>)}}
  clang_analyzer_dump((uchar)(int)x);    // expected-warning{{(unsigned char) (reg_$0<uint x>)}}
  clang_analyzer_dump((uchar)(long)x);   // expected-warning{{(unsigned char) (reg_$0<uint x>)}}
  clang_analyzer_dump((uchar)(llong)x);  // expected-warning{{(unsigned char) (reg_$0<uint x>)}}
  clang_analyzer_dump((uchar)(uchar)x);  // expected-warning{{(unsigned char) (reg_$0<uint x>)}}
  clang_analyzer_dump((uchar)(ushort)x); // expected-warning{{(unsigned char) (reg_$0<uint x>)}}
  clang_analyzer_dump((uchar)(uint)x);   // expected-warning{{(unsigned char) (reg_$0<uint x>)}}
  clang_analyzer_dump((uchar)(ulong)x);  // expected-warning{{(unsigned char) (reg_$0<uint x>)}}
  clang_analyzer_dump((uchar)(ullong)x); // expected-warning{{(unsigned char) (reg_$0<uint x>)}}

  clang_analyzer_dump((ushort)(schar)x);  // expected-warning{{(unsigned short) ((signed char) (reg_$0<uint x>))}}
  clang_analyzer_dump((ushort)(char)x);   // expected-warning{{(unsigned short) ((char) (reg_$0<uint x>))}}
  clang_analyzer_dump((ushort)(short)x);  // expected-warning{{(unsigned short) (reg_$0<uint x>)}}
  clang_analyzer_dump((ushort)(int)x);    // expected-warning{{(unsigned short) (reg_$0<uint x>)}}
  clang_analyzer_dump((ushort)(long)x);   // expected-warning{{(unsigned short) (reg_$0<uint x>)}}
  clang_analyzer_dump((ushort)(llong)x);  // expected-warning{{(unsigned short) (reg_$0<uint x>)}}
  clang_analyzer_dump((ushort)(uchar)x);  // expected-warning{{(unsigned short) ((unsigned char) (reg_$0<uint x>))}}
  clang_analyzer_dump((ushort)(ushort)x); // expected-warning{{(unsigned short) (reg_$0<uint x>)}}
  clang_analyzer_dump((ushort)(uint)x);   // expected-warning{{(unsigned short) (reg_$0<uint x>)}}
  clang_analyzer_dump((ushort)(ulong)x);  // expected-warning{{(unsigned short) (reg_$0<uint x>)}}
  clang_analyzer_dump((ushort)(ullong)x); // expected-warning{{(unsigned short) (reg_$0<uint x>)}}

  clang_analyzer_dump((uint)(schar)x);  // expected-warning{{(signed char) (reg_$0<uint x>)}}
  clang_analyzer_dump((uint)(char)x);   // expected-warning{{(char) (reg_$0<uint x>)}}
  clang_analyzer_dump((uint)(short)x);  // expected-warning{{(short) (reg_$0<uint x>)}}
  clang_analyzer_dump((uint)(int)x);    // expected-warning{{reg_$0<uint x>}}
  clang_analyzer_dump((uint)(long)x);   // expected-warning{{reg_$0<uint x>}}
  clang_analyzer_dump((uint)(llong)x);  // expected-warning{{reg_$0<uint x>}}
  clang_analyzer_dump((uint)(uchar)x);  // expected-warning{{(unsigned char) (reg_$0<uint x>)}}
  clang_analyzer_dump((uint)(ushort)x); // expected-warning{{(unsigned short) (reg_$0<uint x>)}}
  clang_analyzer_dump((uint)(uint)x);   // expected-warning{{reg_$0<uint x>}}
  clang_analyzer_dump((uint)(ulong)x);  // expected-warning{{reg_$0<uint x>}}
  clang_analyzer_dump((uint)(ullong)x); // expected-warning{{reg_$0<uint x>}}

  clang_analyzer_dump((ulong)(schar)x);  // expected-warning{{(unsigned long) ((signed char) (reg_$0<uint x>))}}
  clang_analyzer_dump((ulong)(char)x);   // expected-warning{{(unsigned long) ((char) (reg_$0<uint x>))}}
  clang_analyzer_dump((ulong)(short)x);  // expected-warning{{(unsigned long) ((short) (reg_$0<uint x>))}}
  clang_analyzer_dump((ulong)(int)x);    // expected-warning{{(unsigned long) ((int) (reg_$0<uint x>))}}
  clang_analyzer_dump((ulong)(long)x);   // expected-warning{{(unsigned long) (reg_$0<uint x>)}}
  clang_analyzer_dump((ulong)(llong)x);  // expected-warning{{(unsigned long) (reg_$0<uint x>)}}
  clang_analyzer_dump((ulong)(uchar)x);  // expected-warning{{(unsigned long) ((unsigned char) (reg_$0<uint x>))}}
  clang_analyzer_dump((ulong)(ushort)x); // expected-warning{{(unsigned long) ((unsigned short) (reg_$0<uint x>))}}
  clang_analyzer_dump((ulong)(uint)x);   // expected-warning{{(unsigned long) (reg_$0<uint x>)}}
  clang_analyzer_dump((ulong)(ulong)x);  // expected-warning{{(unsigned long) (reg_$0<uint x>)}}
  clang_analyzer_dump((ulong)(ullong)x); // expected-warning{{(unsigned long) (reg_$0<uint x>)}}

  clang_analyzer_dump((ullong)(schar)x);  // expected-warning{{(unsigned long long) ((signed char) (reg_$0<uint x>))}}
  clang_analyzer_dump((ullong)(char)x);   // expected-warning{{(unsigned long long) ((char) (reg_$0<uint x>))}}
  clang_analyzer_dump((ullong)(short)x);  // expected-warning{{(unsigned long long) ((short) (reg_$0<uint x>))}}
  clang_analyzer_dump((ullong)(int)x);    // expected-warning{{(unsigned long long) ((int) (reg_$0<uint x>))}}
  clang_analyzer_dump((ullong)(long)x);   // expected-warning{{(unsigned long long) (reg_$0<uint x>)}}
  clang_analyzer_dump((ullong)(llong)x);  // expected-warning{{(unsigned long long) (reg_$0<uint x>)}}
  clang_analyzer_dump((ullong)(uchar)x);  // expected-warning{{(unsigned long long) ((unsigned char) (reg_$0<uint x>))}}
  clang_analyzer_dump((ullong)(ushort)x); // expected-warning{{(unsigned long long) ((unsigned short) (reg_$0<uint x>))}}
  clang_analyzer_dump((ullong)(uint)x);   // expected-warning{{(unsigned long long) (reg_$0<uint x>)}}
  clang_analyzer_dump((ullong)(ulong)x);  // expected-warning{{(unsigned long long) (reg_$0<uint x>)}}
  clang_analyzer_dump((ullong)(ullong)x); // expected-warning{{(unsigned long long) (reg_$0<uint x>)}}
}

void test_ulong(ulong x) {
  clang_analyzer_dump(x); // expected-warning{{reg_$0<ulong x>}}

  clang_analyzer_dump((schar)x);  // expected-warning{{(signed char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((char)x);   // expected-warning{{(char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((short)x);  // expected-warning{{(short) (reg_$0<ulong x>)}}
  clang_analyzer_dump((int)x);    // expected-warning{{(int) (reg_$0<ulong x>)}}
  clang_analyzer_dump((long)x);   // expected-warning{{(long) (reg_$0<ulong x>)}}
  clang_analyzer_dump((llong)x);  // expected-warning{{(long long) (reg_$0<ulong x>)}}
  clang_analyzer_dump((uchar)x);  // expected-warning{{(unsigned char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((ushort)x); // expected-warning{{(unsigned short) (reg_$0<ulong x>)}}
  clang_analyzer_dump((uint)x);   // expected-warning{{(unsigned int) (reg_$0<ulong x>)}}
  clang_analyzer_dump((ulong)x);  // expected-warning{{reg_$0<ulong x>}}
  clang_analyzer_dump((ullong)x); // expected-warning{{(unsigned long long) (reg_$0<ulong x>)}}

  clang_analyzer_dump((schar)(schar)x);  // expected-warning{{(signed char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((schar)(char)x);   // expected-warning{{(signed char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((schar)(short)x);  // expected-warning{{(signed char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((schar)(int)x);    // expected-warning{{(signed char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((schar)(long)x);   // expected-warning{{(signed char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((schar)(llong)x);  // expected-warning{{(signed char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((schar)(uchar)x);  // expected-warning{{(signed char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((schar)(ushort)x); // expected-warning{{(signed char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((schar)(uint)x);   // expected-warning{{(signed char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((schar)(ulong)x);  // expected-warning{{(signed char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((schar)(ullong)x); // expected-warning{{(signed char) (reg_$0<ulong x>)}}

  clang_analyzer_dump((char)(schar)x);  // expected-warning{{(char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((char)(char)x);   // expected-warning{{(char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((char)(short)x);  // expected-warning{{(char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((char)(int)x);    // expected-warning{{(char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((char)(long)x);   // expected-warning{{(char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((char)(llong)x);  // expected-warning{{(char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((char)(uchar)x);  // expected-warning{{(char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((char)(ushort)x); // expected-warning{{(char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((char)(uint)x);   // expected-warning{{(char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((char)(ulong)x);  // expected-warning{{(char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((char)(ullong)x); // expected-warning{{(char) (reg_$0<ulong x>)}}

  clang_analyzer_dump((short)(schar)x);  // expected-warning{{(short) ((signed char) (reg_$0<ulong x>))}}
  clang_analyzer_dump((short)(char)x);   // expected-warning{{(short) ((char) (reg_$0<ulong x>))}}
  clang_analyzer_dump((short)(short)x);  // expected-warning{{(short) (reg_$0<ulong x>)}}
  clang_analyzer_dump((short)(int)x);    // expected-warning{{(short) (reg_$0<ulong x>)}}
  clang_analyzer_dump((short)(long)x);   // expected-warning{{(short) (reg_$0<ulong x>)}}
  clang_analyzer_dump((short)(llong)x);  // expected-warning{{(short) (reg_$0<ulong x>)}}
  clang_analyzer_dump((short)(uchar)x);  // expected-warning{{(short) ((unsigned char) (reg_$0<ulong x>))}}
  clang_analyzer_dump((short)(ushort)x); // expected-warning{{(short) (reg_$0<ulong x>)}}
  clang_analyzer_dump((short)(uint)x);   // expected-warning{{(short) (reg_$0<ulong x>)}}
  clang_analyzer_dump((short)(ulong)x);  // expected-warning{{(short) (reg_$0<ulong x>)}}
  clang_analyzer_dump((short)(ullong)x); // expected-warning{{(short) (reg_$0<ulong x>)}}

  clang_analyzer_dump((int)(schar)x);  // expected-warning{{(int) ((signed char) (reg_$0<ulong x>))}}
  clang_analyzer_dump((int)(char)x);   // expected-warning{{(int) ((char) (reg_$0<ulong x>))}}
  clang_analyzer_dump((int)(short)x);  // expected-warning{{(int) ((short) (reg_$0<ulong x>))}}
  clang_analyzer_dump((int)(int)x);    // expected-warning{{(int) (reg_$0<ulong x>)}}
  clang_analyzer_dump((int)(long)x);   // expected-warning{{(int) (reg_$0<ulong x>)}}
  clang_analyzer_dump((int)(llong)x);  // expected-warning{{(int) (reg_$0<ulong x>)}}
  clang_analyzer_dump((int)(uchar)x);  // expected-warning{{(int) ((unsigned char) (reg_$0<ulong x>))}}
  clang_analyzer_dump((int)(ushort)x); // expected-warning{{(int) ((unsigned short) (reg_$0<ulong x>))}}
  clang_analyzer_dump((int)(uint)x);   // expected-warning{{(int) (reg_$0<ulong x>)}}
  clang_analyzer_dump((int)(ulong)x);  // expected-warning{{(int) (reg_$0<ulong x>)}}
  clang_analyzer_dump((int)(ullong)x); // expected-warning{{(int) (reg_$0<ulong x>)}}

  clang_analyzer_dump((long)(schar)x);  // expected-warning{{(long) ((signed char) (reg_$0<ulong x>))}}
  clang_analyzer_dump((long)(char)x);   // expected-warning{{(long) ((char) (reg_$0<ulong x>))}}
  clang_analyzer_dump((long)(short)x);  // expected-warning{{(long) ((short) (reg_$0<ulong x>))}}
  clang_analyzer_dump((long)(int)x);    // expected-warning{{(long) ((int) (reg_$0<ulong x>))}}
  clang_analyzer_dump((long)(long)x);   // expected-warning{{(long) (reg_$0<ulong x>)}}
  clang_analyzer_dump((long)(llong)x);  // expected-warning{{(long) (reg_$0<ulong x>)}}
  clang_analyzer_dump((long)(uchar)x);  // expected-warning{{(long) ((unsigned char) (reg_$0<ulong x>))}}
  clang_analyzer_dump((long)(ushort)x); // expected-warning{{(long) ((unsigned short) (reg_$0<ulong x>))}}
  clang_analyzer_dump((long)(uint)x);   // expected-warning{{(long) ((unsigned int) (reg_$0<ulong x>))}}
  clang_analyzer_dump((long)(ulong)x);  // expected-warning{{(long) (reg_$0<ulong x>)}}
  clang_analyzer_dump((long)(ullong)x); // expected-warning{{(long) (reg_$0<ulong x>)}}

  clang_analyzer_dump((llong)(schar)x);  // expected-warning{{(long long) ((signed char) (reg_$0<ulong x>))}}
  clang_analyzer_dump((llong)(char)x);   // expected-warning{{(long long) ((char) (reg_$0<ulong x>))}}
  clang_analyzer_dump((llong)(short)x);  // expected-warning{{(long long) ((short) (reg_$0<ulong x>))}}
  clang_analyzer_dump((llong)(int)x);    // expected-warning{{(long long) ((int) (reg_$0<ulong x>))}}
  clang_analyzer_dump((llong)(long)x);   // expected-warning{{(long long) (reg_$0<ulong x>)}}
  clang_analyzer_dump((llong)(llong)x);  // expected-warning{{(long long) (reg_$0<ulong x>)}}
  clang_analyzer_dump((llong)(uchar)x);  // expected-warning{{(long long) ((unsigned char) (reg_$0<ulong x>))}}
  clang_analyzer_dump((llong)(ushort)x); // expected-warning{{(long long) ((unsigned short) (reg_$0<ulong x>))}}
  clang_analyzer_dump((llong)(uint)x);   // expected-warning{{(long long) ((unsigned int) (reg_$0<ulong x>))}}
  clang_analyzer_dump((llong)(ulong)x);  // expected-warning{{(long long) (reg_$0<ulong x>)}}
  clang_analyzer_dump((llong)(ullong)x); // expected-warning{{(long long) (reg_$0<ulong x>)}}

  clang_analyzer_dump((uchar)(schar)x);  // expected-warning{{(unsigned char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((uchar)(char)x);   // expected-warning{{(unsigned char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((uchar)(short)x);  // expected-warning{{(unsigned char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((uchar)(int)x);    // expected-warning{{(unsigned char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((uchar)(long)x);   // expected-warning{{(unsigned char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((uchar)(llong)x);  // expected-warning{{(unsigned char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((uchar)(uchar)x);  // expected-warning{{(unsigned char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((uchar)(ushort)x); // expected-warning{{(unsigned char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((uchar)(uint)x);   // expected-warning{{(unsigned char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((uchar)(ulong)x);  // expected-warning{{(unsigned char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((uchar)(ullong)x); // expected-warning{{(unsigned char) (reg_$0<ulong x>)}}

  clang_analyzer_dump((ushort)(schar)x);  // expected-warning{{(unsigned short) ((signed char) (reg_$0<ulong x>))}}
  clang_analyzer_dump((ushort)(char)x);   // expected-warning{{(unsigned short) ((char) (reg_$0<ulong x>))}}
  clang_analyzer_dump((ushort)(short)x);  // expected-warning{{(unsigned short) (reg_$0<ulong x>)}}
  clang_analyzer_dump((ushort)(int)x);    // expected-warning{{(unsigned short) (reg_$0<ulong x>)}}
  clang_analyzer_dump((ushort)(long)x);   // expected-warning{{(unsigned short) (reg_$0<ulong x>)}}
  clang_analyzer_dump((ushort)(llong)x);  // expected-warning{{(unsigned short) (reg_$0<ulong x>)}}
  clang_analyzer_dump((ushort)(uchar)x);  // expected-warning{{(unsigned short) ((unsigned char) (reg_$0<ulong x>))}}
  clang_analyzer_dump((ushort)(ushort)x); // expected-warning{{(unsigned short) (reg_$0<ulong x>)}}
  clang_analyzer_dump((ushort)(uint)x);   // expected-warning{{(unsigned short) (reg_$0<ulong x>)}}
  clang_analyzer_dump((ushort)(ulong)x);  // expected-warning{{(unsigned short) (reg_$0<ulong x>)}}
  clang_analyzer_dump((ushort)(ullong)x); // expected-warning{{(unsigned short) (reg_$0<ulong x>)}}

  clang_analyzer_dump((uint)(schar)x);  // expected-warning{{(unsigned int) ((signed char) (reg_$0<ulong x>))}}
  clang_analyzer_dump((uint)(char)x);   // expected-warning{{(unsigned int) ((char) (reg_$0<ulong x>))}}
  clang_analyzer_dump((uint)(short)x);  // expected-warning{{(unsigned int) ((short) (reg_$0<ulong x>))}}
  clang_analyzer_dump((uint)(int)x);    // expected-warning{{(unsigned int) (reg_$0<ulong x>)}}
  clang_analyzer_dump((uint)(long)x);   // expected-warning{{(unsigned int) (reg_$0<ulong x>)}}
  clang_analyzer_dump((uint)(llong)x);  // expected-warning{{(unsigned int) (reg_$0<ulong x>)}}
  clang_analyzer_dump((uint)(uchar)x);  // expected-warning{{(unsigned int) ((unsigned char) (reg_$0<ulong x>))}}
  clang_analyzer_dump((uint)(ushort)x); // expected-warning{{(unsigned int) ((unsigned short) (reg_$0<ulong x>))}}
  clang_analyzer_dump((uint)(uint)x);   // expected-warning{{(unsigned int) (reg_$0<ulong x>)}}
  clang_analyzer_dump((uint)(ulong)x);  // expected-warning{{(unsigned int) (reg_$0<ulong x>)}}
  clang_analyzer_dump((uint)(ullong)x); // expected-warning{{(unsigned int) (reg_$0<ulong x>)}}

  clang_analyzer_dump((ulong)(schar)x);  // expected-warning{{(signed char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((ulong)(char)x);   // expected-warning{{(char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((ulong)(short)x);  // expected-warning{{(short) (reg_$0<ulong x>)}}
  clang_analyzer_dump((ulong)(int)x);    // expected-warning{{reg_$0<ulong x>}}
  clang_analyzer_dump((ulong)(long)x);   // expected-warning{{reg_$0<ulong x>}}
  clang_analyzer_dump((ulong)(llong)x);  // expected-warning{{reg_$0<ulong x>}}
  clang_analyzer_dump((ulong)(uchar)x);  // expected-warning{{(unsigned char) (reg_$0<ulong x>)}}
  clang_analyzer_dump((ulong)(ushort)x); // expected-warning{{(unsigned short) (reg_$0<ulong x>)}}
  clang_analyzer_dump((ulong)(uint)x);   // expected-warning{{reg_$0<ulong x>}}
  clang_analyzer_dump((ulong)(ulong)x);  // expected-warning{{reg_$0<ulong x>}}
  clang_analyzer_dump((ulong)(ullong)x); // expected-warning{{reg_$0<ulong x>}}

  clang_analyzer_dump((ullong)(schar)x);  // expected-warning{{(unsigned long long) ((signed char) (reg_$0<ulong x>))}}
  clang_analyzer_dump((ullong)(char)x);   // expected-warning{{(unsigned long long) ((char) (reg_$0<ulong x>))}}
  clang_analyzer_dump((ullong)(short)x);  // expected-warning{{(unsigned long long) ((short) (reg_$0<ulong x>))}}
  clang_analyzer_dump((ullong)(int)x);    // expected-warning{{(unsigned long long) ((int) (reg_$0<ulong x>))}}
  clang_analyzer_dump((ullong)(long)x);   // expected-warning{{(unsigned long long) (reg_$0<ulong x>)}}
  clang_analyzer_dump((ullong)(llong)x);  // expected-warning{{(unsigned long long) (reg_$0<ulong x>)}}
  clang_analyzer_dump((ullong)(uchar)x);  // expected-warning{{(unsigned long long) ((unsigned char) (reg_$0<ulong x>))}}
  clang_analyzer_dump((ullong)(ushort)x); // expected-warning{{(unsigned long long) ((unsigned short) (reg_$0<ulong x>))}}
  clang_analyzer_dump((ullong)(uint)x);   // expected-warning{{(unsigned long long) ((unsigned int) (reg_$0<ulong x>))}}
  clang_analyzer_dump((ullong)(ulong)x);  // expected-warning{{(unsigned long long) (reg_$0<ulong x>)}}
  clang_analyzer_dump((ullong)(ullong)x); // expected-warning{{(unsigned long long) (reg_$0<ulong x>)}}
}

void test_llong(ullong x) {
  clang_analyzer_dump(x); // expected-warning{{reg_$0<ullong x>}}

  clang_analyzer_dump((schar)x);  // expected-warning{{(signed char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((char)x);   // expected-warning{{(char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((short)x);  // expected-warning{{(short) (reg_$0<ullong x>)}}
  clang_analyzer_dump((int)x);    // expected-warning{{(int) (reg_$0<ullong x>)}}
  clang_analyzer_dump((long)x);   // expected-warning{{(long) (reg_$0<ullong x>)}}
  clang_analyzer_dump((llong)x);  // expected-warning{{(long long) (reg_$0<ullong x>)}}
  clang_analyzer_dump((uchar)x);  // expected-warning{{(unsigned char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((ushort)x); // expected-warning{{(unsigned short) (reg_$0<ullong x>)}}
  clang_analyzer_dump((uint)x);   // expected-warning{{(unsigned int) (reg_$0<ullong x>)}}
  clang_analyzer_dump((ulong)x);  // expected-warning{{(unsigned long) (reg_$0<ullong x>)}}
  clang_analyzer_dump((ullong)x); // expected-warning{{reg_$0<ullong x>}}

  clang_analyzer_dump((schar)(schar)x);  // expected-warning{{(signed char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((schar)(char)x);   // expected-warning{{(signed char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((schar)(short)x);  // expected-warning{{(signed char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((schar)(int)x);    // expected-warning{{(signed char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((schar)(long)x);   // expected-warning{{(signed char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((schar)(llong)x);  // expected-warning{{(signed char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((schar)(uchar)x);  // expected-warning{{(signed char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((schar)(ushort)x); // expected-warning{{(signed char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((schar)(uint)x);   // expected-warning{{(signed char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((schar)(ulong)x);  // expected-warning{{(signed char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((schar)(ullong)x); // expected-warning{{(signed char) (reg_$0<ullong x>)}}

  clang_analyzer_dump((char)(schar)x);  // expected-warning{{(char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((char)(char)x);   // expected-warning{{(char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((char)(short)x);  // expected-warning{{(char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((char)(int)x);    // expected-warning{{(char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((char)(long)x);   // expected-warning{{(char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((char)(llong)x);  // expected-warning{{(char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((char)(uchar)x);  // expected-warning{{(char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((char)(ushort)x); // expected-warning{{(char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((char)(uint)x);   // expected-warning{{(char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((char)(ulong)x);  // expected-warning{{(char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((char)(ullong)x); // expected-warning{{(char) (reg_$0<ullong x>)}}

  clang_analyzer_dump((short)(schar)x);  // expected-warning{{(short) ((signed char) (reg_$0<ullong x>))}}
  clang_analyzer_dump((short)(char)x);   // expected-warning{{(short) ((char) (reg_$0<ullong x>))}}
  clang_analyzer_dump((short)(short)x);  // expected-warning{{(short) (reg_$0<ullong x>)}}
  clang_analyzer_dump((short)(int)x);    // expected-warning{{(short) (reg_$0<ullong x>)}}
  clang_analyzer_dump((short)(long)x);   // expected-warning{{(short) (reg_$0<ullong x>)}}
  clang_analyzer_dump((short)(llong)x);  // expected-warning{{(short) (reg_$0<ullong x>)}}
  clang_analyzer_dump((short)(uchar)x);  // expected-warning{{(short) ((unsigned char) (reg_$0<ullong x>))}}
  clang_analyzer_dump((short)(ushort)x); // expected-warning{{(short) (reg_$0<ullong x>)}}
  clang_analyzer_dump((short)(uint)x);   // expected-warning{{(short) (reg_$0<ullong x>)}}
  clang_analyzer_dump((short)(ulong)x);  // expected-warning{{(short) (reg_$0<ullong x>)}}
  clang_analyzer_dump((short)(ullong)x); // expected-warning{{(short) (reg_$0<ullong x>)}}

  clang_analyzer_dump((int)(schar)x);  // expected-warning{{(int) ((signed char) (reg_$0<ullong x>))}}
  clang_analyzer_dump((int)(char)x);   // expected-warning{{(int) ((char) (reg_$0<ullong x>))}}
  clang_analyzer_dump((int)(short)x);  // expected-warning{{(int) ((short) (reg_$0<ullong x>))}}
  clang_analyzer_dump((int)(int)x);    // expected-warning{{(int) (reg_$0<ullong x>)}}
  clang_analyzer_dump((int)(long)x);   // expected-warning{{(int) (reg_$0<ullong x>)}}
  clang_analyzer_dump((int)(llong)x);  // expected-warning{{(int) (reg_$0<ullong x>)}}
  clang_analyzer_dump((int)(uchar)x);  // expected-warning{{(int) ((unsigned char) (reg_$0<ullong x>))}}
  clang_analyzer_dump((int)(ushort)x); // expected-warning{{(int) ((unsigned short) (reg_$0<ullong x>))}}
  clang_analyzer_dump((int)(uint)x);   // expected-warning{{(int) (reg_$0<ullong x>)}}
  clang_analyzer_dump((int)(ulong)x);  // expected-warning{{(int) (reg_$0<ullong x>)}}
  clang_analyzer_dump((int)(ullong)x); // expected-warning{{(int) (reg_$0<ullong x>)}}

  clang_analyzer_dump((long)(schar)x);  // expected-warning{{(long) ((signed char) (reg_$0<ullong x>))}}
  clang_analyzer_dump((long)(char)x);   // expected-warning{{(long) ((char) (reg_$0<ullong x>))}}
  clang_analyzer_dump((long)(short)x);  // expected-warning{{(long) ((short) (reg_$0<ullong x>))}}
  clang_analyzer_dump((long)(int)x);    // expected-warning{{(long) ((int) (reg_$0<ullong x>))}}
  clang_analyzer_dump((long)(long)x);   // expected-warning{{(long) (reg_$0<ullong x>)}}
  clang_analyzer_dump((long)(llong)x);  // expected-warning{{(long) (reg_$0<ullong x>)}}
  clang_analyzer_dump((long)(uchar)x);  // expected-warning{{(long) ((unsigned char) (reg_$0<ullong x>))}}
  clang_analyzer_dump((long)(ushort)x); // expected-warning{{(long) ((unsigned short) (reg_$0<ullong x>))}}
  clang_analyzer_dump((long)(uint)x);   // expected-warning{{(long) ((unsigned int) (reg_$0<ullong x>))}}
  clang_analyzer_dump((long)(ulong)x);  // expected-warning{{(long) (reg_$0<ullong x>)}}
  clang_analyzer_dump((long)(ullong)x); // expected-warning{{(long) (reg_$0<ullong x>)}}

  clang_analyzer_dump((llong)(schar)x);  // expected-warning{{(long long) ((signed char) (reg_$0<ullong x>))}}
  clang_analyzer_dump((llong)(char)x);   // expected-warning{{(long long) ((char) (reg_$0<ullong x>))}}
  clang_analyzer_dump((llong)(short)x);  // expected-warning{{(long long) ((short) (reg_$0<ullong x>))}}
  clang_analyzer_dump((llong)(int)x);    // expected-warning{{(long long) ((int) (reg_$0<ullong x>))}}
  clang_analyzer_dump((llong)(long)x);   // expected-warning{{(long long) (reg_$0<ullong x>)}}
  clang_analyzer_dump((llong)(llong)x);  // expected-warning{{(long long) (reg_$0<ullong x>)}}
  clang_analyzer_dump((llong)(uchar)x);  // expected-warning{{(long long) ((unsigned char) (reg_$0<ullong x>))}}
  clang_analyzer_dump((llong)(ushort)x); // expected-warning{{(long long) ((unsigned short) (reg_$0<ullong x>))}}
  clang_analyzer_dump((llong)(uint)x);   // expected-warning{{(long long) ((unsigned int) (reg_$0<ullong x>))}}
  clang_analyzer_dump((llong)(ulong)x);  // expected-warning{{(long long) (reg_$0<ullong x>)}}
  clang_analyzer_dump((llong)(ullong)x); // expected-warning{{(long long) (reg_$0<ullong x>)}}

  clang_analyzer_dump((uchar)(schar)x);  // expected-warning{{(unsigned char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((uchar)(char)x);   // expected-warning{{(unsigned char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((uchar)(short)x);  // expected-warning{{(unsigned char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((uchar)(int)x);    // expected-warning{{(unsigned char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((uchar)(long)x);   // expected-warning{{(unsigned char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((uchar)(llong)x);  // expected-warning{{(unsigned char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((uchar)(uchar)x);  // expected-warning{{(unsigned char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((uchar)(ushort)x); // expected-warning{{(unsigned char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((uchar)(uint)x);   // expected-warning{{(unsigned char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((uchar)(ulong)x);  // expected-warning{{(unsigned char) (reg_$0<ullong x>)}}
  clang_analyzer_dump((uchar)(ullong)x); // expected-warning{{(unsigned char) (reg_$0<ullong x>)}}

  clang_analyzer_dump((ushort)(schar)x);  // expected-warning{{(unsigned short) ((signed char) (reg_$0<ullong x>))}}
  clang_analyzer_dump((ushort)(char)x);   // expected-warning{{(unsigned short) ((char) (reg_$0<ullong x>))}}
  clang_analyzer_dump((ushort)(short)x);  // expected-warning{{(unsigned short) (reg_$0<ullong x>)}}
  clang_analyzer_dump((ushort)(int)x);    // expected-warning{{(unsigned short) (reg_$0<ullong x>)}}
  clang_analyzer_dump((ushort)(long)x);   // expected-warning{{(unsigned short) (reg_$0<ullong x>)}}
  clang_analyzer_dump((ushort)(llong)x);  // expected-warning{{(unsigned short) (reg_$0<ullong x>)}}
  clang_analyzer_dump((ushort)(uchar)x);  // expected-warning{{(unsigned short) ((unsigned char) (reg_$0<ullong x>))}}
  clang_analyzer_dump((ushort)(ushort)x); // expected-warning{{(unsigned short) (reg_$0<ullong x>)}}
  clang_analyzer_dump((ushort)(uint)x);   // expected-warning{{(unsigned short) (reg_$0<ullong x>)}}
  clang_analyzer_dump((ushort)(ulong)x);  // expected-warning{{(unsigned short) (reg_$0<ullong x>)}}
  clang_analyzer_dump((ushort)(ullong)x); // expected-warning{{(unsigned short) (reg_$0<ullong x>)}}

  clang_analyzer_dump((uint)(schar)x);  // expected-warning{{(unsigned int) ((signed char) (reg_$0<ullong x>))}}
  clang_analyzer_dump((uint)(char)x);   // expected-warning{{(unsigned int) ((char) (reg_$0<ullong x>))}}
  clang_analyzer_dump((uint)(short)x);  // expected-warning{{(unsigned int) ((short) (reg_$0<ullong x>))}}
  clang_analyzer_dump((uint)(int)x);    // expected-warning{{(unsigned int) (reg_$0<ullong x>)}}
  clang_analyzer_dump((uint)(long)x);   // expected-warning{{(unsigned int) (reg_$0<ullong x>)}}
  clang_analyzer_dump((uint)(llong)x);  // expected-warning{{(unsigned int) (reg_$0<ullong x>)}}
  clang_analyzer_dump((uint)(uchar)x);  // expected-warning{{(unsigned int) ((unsigned char) (reg_$0<ullong x>))}}
  clang_analyzer_dump((uint)(ushort)x); // expected-warning{{(unsigned int) ((unsigned short) (reg_$0<ullong x>))}}
  clang_analyzer_dump((uint)(uint)x);   // expected-warning{{(unsigned int) (reg_$0<ullong x>)}}
  clang_analyzer_dump((uint)(ulong)x);  // expected-warning{{(unsigned int) (reg_$0<ullong x>)}}
  clang_analyzer_dump((uint)(ullong)x); // expected-warning{{(unsigned int) (reg_$0<ullong x>)}}

  clang_analyzer_dump((ulong)(schar)x);  // expected-warning{{(unsigned long) ((signed char) (reg_$0<ullong x>))}}
  clang_analyzer_dump((ulong)(char)x);   // expected-warning{{(unsigned long) ((char) (reg_$0<ullong x>))}}
  clang_analyzer_dump((ulong)(short)x);  // expected-warning{{(unsigned long) ((short) (reg_$0<ullong x>))}}
  clang_analyzer_dump((ulong)(int)x);    // expected-warning{{(unsigned long) ((int) (reg_$0<ullong x>))}}
  clang_analyzer_dump((ulong)(long)x);   // expected-warning{{(unsigned long) (reg_$0<ullong x>)}}
  clang_analyzer_dump((ulong)(llong)x);  // expected-warning{{(unsigned long) (reg_$0<ullong x>)}}
  clang_analyzer_dump((ulong)(uchar)x);  // expected-warning{{(unsigned long) ((unsigned char) (reg_$0<ullong x>))}}
  clang_analyzer_dump((ulong)(ushort)x); // expected-warning{{(unsigned long) ((unsigned short) (reg_$0<ullong x>))}}
  clang_analyzer_dump((ulong)(uint)x);   // expected-warning{{(unsigned long) ((unsigned int) (reg_$0<ullong x>))}}
  clang_analyzer_dump((ulong)(ulong)x);  // expected-warning{{(unsigned long) (reg_$0<ullong x>)}}
  clang_analyzer_dump((ulong)(ullong)x); // expected-warning{{(unsigned long) (reg_$0<ullong x>)}}

  clang_analyzer_dump((ullong)(schar)x);  // expected-warning{{(unsigned long long) ((signed char) (reg_$0<ullong x>))}}
  clang_analyzer_dump((ullong)(char)x);   // expected-warning{{(unsigned long long) ((char) (reg_$0<ullong x>))}}
  clang_analyzer_dump((ullong)(short)x);  // expected-warning{{(unsigned long long) ((short) (reg_$0<ullong x>))}}
  clang_analyzer_dump((ullong)(int)x);    // expected-warning{{(unsigned long long) ((int) (reg_$0<ullong x>))}}
  clang_analyzer_dump((ullong)(long)x);   // expected-warning{{reg_$0<ullong x>}}
  clang_analyzer_dump((ullong)(llong)x);  // expected-warning{{reg_$0<ullong x>}}
  clang_analyzer_dump((ullong)(uchar)x);  // expected-warning{{(unsigned long long) ((unsigned char) (reg_$0<ullong x>))}}
  clang_analyzer_dump((ullong)(ushort)x); // expected-warning{{(unsigned long long) ((unsigned short) (reg_$0<ullong x>))}}
  clang_analyzer_dump((ullong)(uint)x);   // expected-warning{{(unsigned long long) ((unsigned int) (reg_$0<ullong x>))}}
  clang_analyzer_dump((ullong)(ulong)x);  // expected-warning{{reg_$0<ullong x>}}
  clang_analyzer_dump((ullong)(ullong)x); // expected-warning{{reg_$0<ullong x>}}
}
