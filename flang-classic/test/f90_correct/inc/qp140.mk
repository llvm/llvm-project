#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test iqnint  ########


qp140: run
	

build:  $(SRC)/qp140.f08
	-$(RM) qp140.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp140.f08 -o qp140.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp140.$(OBJX) check.$(OBJX) $(LIBS) -o qp140.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test qp140 
	qp140.$(EXESUFFIX)

verify: ;


