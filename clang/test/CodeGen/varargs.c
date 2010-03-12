// RUN: %clang_cc1 -emit-llvm -o - %s


// PR6433 - Don't crash on va_arg(typedef).
typedef double gdouble;
void focus_changed_cb () {
    __builtin_va_list pa;
    double mfloat;
    mfloat = __builtin_va_arg((pa), gdouble);
}

