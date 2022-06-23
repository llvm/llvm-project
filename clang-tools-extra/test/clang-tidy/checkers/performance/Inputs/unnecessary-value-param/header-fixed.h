// struct ABC is expensive to copy and should be
// passed as a const reference.
struct ABC {
  ABC(const ABC&);
  int get(int) const;
};


int f1(int n,              const ABC& v1,   const ABC& v2); // line 9

int f1(int n, ABC v1); // line 11



int f2(        int n,       const ABC& v2); // line 15
