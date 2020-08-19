"""
This module builds test binaries for the test suite using Make.

Platform specific builders can override methods in the Builder base class. The
factory method below hands out builders based on the given platform.
"""

from .builder import Builder


def get_builder(platform):
  """Returns a Builder instance for the given platform."""
  if platform == 'darwin':
    from .darwin import BuilderDarwin
    return BuilderDarwin()
  return Builder()
