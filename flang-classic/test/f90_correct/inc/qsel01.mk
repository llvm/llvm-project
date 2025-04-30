#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qsel01  ########


qsel01: run
	

build:  $(SRC)/qsel01.f08
	-$(RM) qsel01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qsel01.f08 -o qsel01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qsel01.$(OBJX) check.$(OBJX) $(LIBS) -o qsel01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qsel01
	qsel01.$(EXESUFFIX)

verify: ;

qsel01.run: run

