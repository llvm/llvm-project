// test case
#define NULL 0
void* AllocBuff();
void Log1();
void Log2();

int GetData(int *data)
{
	return data[0];
}

int *allocMem()
{
    int *rbd = (int *)AllocBuff();
    if (rbd != NULL) {
        Log1();
    } else {
        Log2();
    }
    return rbd;
}

void func1(int *outdata1, int *outdata2)
{
    int *rbd = NULL;
    int loop;

    for (loop = 0; loop < 5; loop++) {
        rbd = allocMem();
        if(1) {}

		*outdata1 = GetData(rbd);
		Log1();
		*outdata2 = GetData(rbd);
    }
}