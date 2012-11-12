// RUN: %clang -fverbose-asm -S -g %s -o - | grep DW_TAG_enumeration_type

int v;
enum index  { MAX };
void foo(void)
{
  v = MAX;
}
