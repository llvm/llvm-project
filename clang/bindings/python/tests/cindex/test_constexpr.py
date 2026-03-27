import unittest
from .util import get_cursor, get_tu


class TestConstexpr(unittest.TestCase):
    def test_is_constexpr(self):
        source = """
        constexpr int f1() {
          constexpr int local_v1 = 1;
          int local_v2 = 2;
          return local_v1 + local_v2;
        }
        int f2() { return 2; }
        
        constexpr int v1 = 3;
        int v2 = 4;
        
        struct S {
          static constexpr int m1 = 5;
          int m2;
          constexpr int m3() const { return 6; }
          int m4() const { return 7; }
        };
        """
        tu = get_tu(source, lang="cpp")

        f1 = get_cursor(tu, "f1")
        f2 = get_cursor(tu, "f2")
        self.assertTrue(f1.is_constexpr)
        self.assertFalse(f2.is_constexpr)

        v1 = get_cursor(tu, "v1")
        v2 = get_cursor(tu, "v2")
        self.assertTrue(v1.is_constexpr)
        self.assertFalse(v2.is_constexpr)

        local_v1 = get_cursor(f1, "local_v1")
        local_v2 = get_cursor(f1, "local_v2")
        self.assertTrue(local_v1.is_constexpr)
        self.assertFalse(local_v2.is_constexpr)

        S = get_cursor(tu, "S")
        m1 = get_cursor(S, "m1")
        m2 = get_cursor(S, "m2")
        m3 = get_cursor(S, "m3")
        m4 = get_cursor(S, "m4")

        self.assertTrue(m1.is_constexpr)
        self.assertFalse(m2.is_constexpr)
        self.assertTrue(m3.is_constexpr)
        self.assertFalse(m4.is_constexpr)
