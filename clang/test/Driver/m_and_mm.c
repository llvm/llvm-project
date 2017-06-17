// RUN: %clang -### \
// RUN:   -M -MM %s 2> %t
// RUN: not grep '"-sys-header-deps"' %t

// RUN: %clang -M -MM %s 2> %t
// RUN: not grep "warning" %t

// RUN: %clang -MMD -MD %s 2> %t || true
// RUN: grep "warning" %t

#warning "This warning shouldn't show up with -M and -MM"
int main (void)
{
    return 0;
}
