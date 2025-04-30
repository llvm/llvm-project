#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn05a  ########


qn05a: run
	

build:  $(SRC)/qn05a.f90
	-$(RM) qn05a.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qn05a.f90 -o qn05a.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qn05a.$(OBJX) check.$(OBJX) $(LIBS) -o qn05a.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn05a
	qn05a.$(EXESUFFIX)

verify: ;

qn05a.run: run

