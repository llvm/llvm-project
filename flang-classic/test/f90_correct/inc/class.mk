#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test class ########


class: run

build:  $(SRC)/class.f90
	-$(RM) class.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/class.f90 -o class.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) class.$(OBJX) $(LIBS) -o class.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test class
	class.$(EXESUFFIX)

verify: ;

class.run: run

