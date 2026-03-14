export module bar;
import Templ;
export template<class T>
int bar() {
    return G<T>()();
}
