# This file is included several times in a row, once
# for each element of $(object-suffixes).  $(object-suffixes-left)
# is initialized first to $(object-suffixes) so that with each
# inclusion, we advance $o to the next suffix.

o := $(firstword $(object-suffixes-left))
object-suffixes-left := $(filter-out $o,$(object-suffixes-left))

$(o-iterator-doit)
