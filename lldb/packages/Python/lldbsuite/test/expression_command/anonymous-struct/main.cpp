#include <tgmath.h>

typedef struct {
    float f;
    int i;
} my_untagged_struct;

double multiply(my_untagged_struct *s)
{
    return s->f * s->i;
}

double multiply(my_untagged_struct *s, int x)
{
    return multiply(s) * x;
}

int main(int argc, char **argv)
{
    my_untagged_struct s = {
        .f = (float)argc,
        .i = argc,
    };
    // lldb testsuite break
    return !(multiply(&s, argc) == pow(argc, 3));
}
