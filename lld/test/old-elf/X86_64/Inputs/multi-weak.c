int fn()
{
  return 0;
}

void __attribute__((weak)) f()
{
        printf("original f..\n");
}

void __attribute__((weak)) g()
{
        printf("original f..\n");
}

int main(void)
{
        f();
        return 0;
}
