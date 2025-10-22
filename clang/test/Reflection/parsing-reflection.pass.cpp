// RUN: %clang_cc1 %s -std=c++23 -freflection


int main()
{
    (void)(^^::);
    (void)(^^void);
    (void)(^^bool);
    (void)(^^char);
    (void)(^^signed char);
    (void)(^^unsigned char);
    (void)(^^short);
    (void)(^^unsigned short);
    (void)(^^int);
    (void)(^^unsigned int);
    (void)(^^long);
    (void)(^^unsigned long);
    (void)(^^long long);
    (void)(^^float);
    (void)(^^double);
}
