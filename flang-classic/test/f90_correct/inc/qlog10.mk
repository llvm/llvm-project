#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qlog10  ########


qlog10: run
	

build:  $(SRC)/qlog10.f08
	-$(RM) qlog10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qlog10.f08 -o qlog10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qlog10.$(OBJX) check.$(OBJX) $(LIBS) -o qlog10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qlog10
	qlog10.$(EXESUFFIX)

verify: ;

qlog10.run: run

