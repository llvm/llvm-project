class Foo
{
public:
    int Bar(int x, int y)
    {
        return x + y;
    }
};

struct Container { int MemberVar; };

int main()
{
    Foo fooo;
    Foo *ptr_fooo = &fooo;
    fooo.Bar(1, 2);

    Container container;
    Container *ptr_container = &container;
    return container.MemberVar = 3; // Break here
}
