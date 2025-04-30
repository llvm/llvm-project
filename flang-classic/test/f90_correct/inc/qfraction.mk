#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qfraction  ########


qfraction: run
	

build:  $(SRC)/qfraction.f08
	-$(RM) qfraction.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qfraction.f08 -o qfraction.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qfraction.$(OBJX) check.$(OBJX) $(LIBS) -o qfraction.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qfraction
	qfraction.$(EXESUFFIX)

verify: ;

qfraction.run: run

