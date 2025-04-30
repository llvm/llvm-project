#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qint  ########


qint: run
	

build:  $(SRC)/qint.f08
	-$(RM) qint.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qint.f08 -o qint.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qint.$(OBJX) check.$(OBJX) $(LIBS) -o qint.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test qint 
	qint.$(EXESUFFIX)

verify: ;


