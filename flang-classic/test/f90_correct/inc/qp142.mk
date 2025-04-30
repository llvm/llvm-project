#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test iqnint  ########


qp142: run
	

build:  $(SRC)/qp142.f08
	-$(RM) qp142.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp142.f08 -o qp142.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp142.$(OBJX) check.$(OBJX) $(LIBS) -o qp142.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test qp142 
	qp142.$(EXESUFFIX)

verify: ;


