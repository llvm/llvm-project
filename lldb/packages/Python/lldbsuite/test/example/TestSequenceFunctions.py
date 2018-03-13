"""An example unittest copied from python tutorial."""

import random
import unittest
import traceback


class SequenceFunctionsTestCase(unittest.TestCase):

    def setUp(self):
        # traceback.print_stack()
        self.seq = list(range(10))

    def tearDown(self):
        # traceback.print_stack()
        pass

    def test_shuffle(self):
        # make sure the shuffled sequence does not lose any elements
        random.shuffle(self.seq)
        self.seq.sort()
        self.assertEqual(self.seq, list(range(10)))

    def test_choice(self):
        element = random.choice(self.seq)
        self.assertTrue(element in self.seq)

    def test_sample(self):
        self.assertRaises(ValueError, random.sample, self.seq, 20)
        for element in random.sample(self.seq, 5):
            self.assertTrue(element in self.seq)

if __name__ == '__main__':
    unittest.main()
