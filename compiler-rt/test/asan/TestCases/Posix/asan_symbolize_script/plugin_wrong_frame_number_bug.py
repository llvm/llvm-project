import logging

class FailOncePlugin(AsanSymbolizerPlugIn):
  """
    This is a simple plug-in that always claims
    that a binary can't be symbolized on the first
    call but succeeds for all subsequent calls.

    This plug-in exists to reproduce an old bug
    in the `asan_symbolize.py` script.

    By failing the first symbolization request
    we used to cause an early exit in `asan_symbolize.py`
    that didn't increment the frame counter which
    caused subsequent symbolization attempts to
    print the wrong frame number.
  """
  def __init__(self):
    self.should_fail = True
    pass

  def filter_binary_path(self, path):
    logging.info('filter_binary_path called in NoOpPlugin')
    if self.should_fail:
      logging.info('Doing first fail')
      self.should_fail = False
      return None
    logging.info('Doing succeed')
    return path

register_plugin(FailOncePlugin())
