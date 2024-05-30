// test case

void* AllocBuff();
void Func_2(int *data);
int Func_5(int num);
extern int err_code;
static inline void Log()
{
}

int Func_3(int *data)
{
    if (data[0] > 10) {
	    Log();
		return err_code;
	}
	
	Func_2(data);
	return 0;
}

int Func_4(int *data)
{
    return Func_5(*data);
}

int Func_Proc(int *data)
{
	int ret;
	ret = Func_3(data);
	if (ret != 0) {
	    return ret;
	}
	
	ret = Func_4(data);
	if (ret != 0) {
	    return ret;
	}
	
	Func_2(data);
}

int Func_1()
{
    int ret;
    int *data = (int *)AllocBuff();
	
	ret = Func_Proc(data);
	if (ret == 1) {
	    Func_2(data);
	} else if (ret != 0) {
		Log();
		return -1;
	}
	
	Func_2(data);
}
