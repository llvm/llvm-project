// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

void schedule_timeout(signed long timeout)
{
 switch (timeout)
 {
 case ((long)(~0UL>>1)): break;
 }
}
