#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp05  ########


qp05: run


build:  $(SRC)/qp05.f08
	-$(RM) qp05.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp05.f08 -o qp05.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp05.$(OBJX) check.$(OBJX) $(LIBS) -o qp05.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp05
	qp05.$(EXESUFFIX)

verify: ;
