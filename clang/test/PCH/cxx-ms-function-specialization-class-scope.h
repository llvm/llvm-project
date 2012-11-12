


template <class T>
class B {
public:
	template <class U>
    B(U p) { 
	}
	template <>
    B(int p) { // expected-warning{{explicit specialization of 'B<T>' within class scope is a Microsoft extension}}
	}
	
	template <class U>
    void f(U p) {
	  T y = 9;
	}


    template <>
    void f(int p) { // expected-warning{{explicit specialization of 'f' within class scope is a Microsoft extension}}
	  T a = 3;
	}

	void f(int p) { 
 	  T a = 3;
    }
};

