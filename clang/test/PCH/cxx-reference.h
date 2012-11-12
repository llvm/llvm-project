// Header for PCH test cxx-reference.cpp

typedef char (&LR);
typedef char (&&RR);

char c;

char &lr = c;
char &&rr = 'c';
LR &lrlr = c;
LR &&rrlr = c;
RR &lrrr = c;
RR &&rrrr = 'c';
