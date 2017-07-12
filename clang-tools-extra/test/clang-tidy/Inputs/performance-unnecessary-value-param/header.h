// struct ABC is expensive to copy and should be
// passed as a const referece.
struct ABC {
  ABC(const ABC&);
  int get(int) const;
};


int f1(int n,              ABC v1,   ABC v2); // line 9

int f1(int n, ABC v1); // line 11



int f2(        int n,       ABC v2); // line 15
