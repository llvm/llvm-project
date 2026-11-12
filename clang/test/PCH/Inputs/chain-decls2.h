void g(void);

struct two {};
void one(void);
struct three {}; // for verification

void many(int k);
struct many;
void many(int l);
struct many {};

void noret(void) __attribute__((noreturn));
