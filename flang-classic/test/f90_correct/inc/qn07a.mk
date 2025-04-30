#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn07a  ########


qn07a: run
	

build:  $(SRC)/qn07a.f90
	-$(RM) qn07a.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qn07a.f90 -o qn07a.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qn07a.$(OBJX) check.$(OBJX) $(LIBS) -o qn07a.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn07a
	qn07a.$(EXESUFFIX)

verify: ;

qn07a.run: run

