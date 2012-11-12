void g();

struct two {};
void one();
struct three {}; // for verification

void many(int k);
struct many;
void many(int l);
struct many {};

void noret() __attribute__((noreturn));
