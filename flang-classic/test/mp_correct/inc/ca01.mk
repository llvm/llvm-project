#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
ca01: ca01.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN2) ./a.$(EXESUFFIX) $(LOG)
ca01.$(OBJX): $(SRC)/ca01.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/ca01.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) ca01.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: ca01
run: ;
