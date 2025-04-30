#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn07  ########


qn07: run
	

build:  $(SRC)/qn07.f90
	-$(RM) qn07.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qn07.f90 -o qn07.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qn07.$(OBJX) check.$(OBJX) $(LIBS) -o qn07.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn07
	qn07.$(EXESUFFIX)

verify: ;

qn07.run: run

