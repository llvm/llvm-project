#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test iqnint  ########


qp138: run
	

build:  $(SRC)/qp138.f08
	-$(RM) qp138.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp138.f08 -o qp138.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp138.$(OBJX) check.$(OBJX) $(LIBS) -o qp138.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test qp138 
	qp138.$(EXESUFFIX)

verify: ;


