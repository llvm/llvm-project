typedef struct {
    int a;
    int b;
} tempStru;

tempStru g_var;
tempStru *getPrt(int flag)
{
    if (flag == 0) {
        return nullptr; // src
    }
    return &g_var;
}

bool g_array[10] = {0};
bool isOpen(int flag)
{
    return g_array[flag];
}

void usePtr2(tempStru *p, int flag)
{
    if (isOpen(flag) != true) {
        return;
    }

    p->a = 1; // sink
    p->b = 2;
}

void func1(int flag)
{
    tempStru *p = getPrt(flag); // point1
    usePtr2(p, 0); // point2
}


void func2(int flag)
{
    func1(flag);
}

void func3(int flag)
{
    func2(flag);
}


bool func4(int flag)
{
    func3(flag);

    if (isOpen(flag) == true) {
        return true;
    }
    return false;
}
