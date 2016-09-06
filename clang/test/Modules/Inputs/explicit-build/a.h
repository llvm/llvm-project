#if !__building_module(a) && !BUILDING_A_PCH
#error "should only get here when building module a"
#endif

const int a = 1;
