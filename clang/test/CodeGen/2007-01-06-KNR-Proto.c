// RUN: %clang_cc1 -emit-llvm -o - %s
// PR1083

int svc_register (void (*dispatch) (int));

int svc_register (dispatch)
     void (*dispatch) ();
{
}

