// Template instantiations are placed into comdat sections. Check that
// coverage data from different instantiations are mapped back to the correct
// source regions.

template <class T> class FOO {
public:
  FOO() : t(0) {}

  T DoIt(T ti);

private:
  T t;
};

template <class T> T FOO<T>::DoIt(T ti) { // HEADER:  2| [[@LINE]]|template
  for (T I = 0; I < ti; I++) {            // HEADER: 22| [[@LINE]]|  for (T
    t += I;                               // HEADER: 20| [[@LINE]]|    t += I;
    if (I > ti / 2)                       // HEADER: 20| [[@LINE]]|    if (I > ti 
      t -= 1;                             // HEADER:  8| [[@LINE]]|      t -= 1;
  }                                       // HEADER: 10| [[@LINE]]|  }
                                          // HEADER:  1| [[@LINE]]|
  return t;                               // HEADER:  1| [[@LINE]]|  return t;
}
