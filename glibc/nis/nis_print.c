/* Copyright (c) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Thorsten Kukuk <kukuk@vt.uni-paderborn.de>, 1997.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <time.h>
#include <string.h>
#include <libintl.h>
#include <stdint.h>

#include <rpcsvc/nis.h>
#include <shlib-compat.h>

static const char *
nis_nstype2str (const nstype type)
{

/* Name service names mustn't be translated, only UNKNOWN needs it */

  switch (type)
    {
    case NIS:
      return "NIS";
    case SUNYP:
      return "SUNYP";
    case IVY:
      return "IVY";
    case DNS:
      return "DNS";
    case X500:
      return "X500";
    case DNANS:
      return "DNANS";
    case XCHS:
      return "XCHS";
    case CDS:
      return "CDS";
    default:
      return N_("UNKNOWN");
    }
}

static void
print_ttl (const uint32_t ttl)
{
  uint32_t time, s, m, h;

  time = ttl;

  h = time / (60 * 60);
  time %= (60 * 60);
  m = time / 60;
  time %= 60;
  s = time;
  printf ("%u:%u:%u\n", h, m, s);
}

static void
print_flags (const unsigned int flags)
{
  fputs ("(", stdout);

  if (flags & TA_SEARCHABLE)
    fputs ("SEARCHABLE, ", stdout);

  if (flags & TA_BINARY)
    {
      fputs ("BINARY DATA", stdout);
      if (flags & TA_XDR)
	fputs (", XDR ENCODED", stdout);
      if (flags & TA_ASN1)
	fputs (", ASN.1 ENCODED", stdout);
      if (flags & TA_CRYPT)
	fputs (", ENCRYPTED", stdout);
    }
  else
    {
      fputs ("TEXTUAL DATA", stdout);
      if (flags & TA_SEARCHABLE)
	{
	  if (flags & TA_CASE)
	    fputs (", CASE INSENSITIVE", stdout);
	  else
	    fputs (", CASE SENSITIVE", stdout);
	}
    }

  fputs (")\n", stdout);
}

static void
nis_print_objtype (enum zotypes type)
{
  switch (type)
    {
    case NIS_BOGUS_OBJ:
      fputs (_("BOGUS OBJECT\n"), stdout);
      break;
    case NIS_NO_OBJ:
      fputs (_("NO OBJECT\n"), stdout);
      break;
    case NIS_DIRECTORY_OBJ:
      fputs (_("DIRECTORY\n"), stdout);
      break;
    case NIS_GROUP_OBJ:
      fputs (_("GROUP\n"), stdout);
      break;
    case NIS_TABLE_OBJ:
      fputs (_("TABLE\n"), stdout);
      break;
    case NIS_ENTRY_OBJ:
      fputs (_("ENTRY\n"), stdout);
      break;
    case NIS_LINK_OBJ:
      fputs (_("LINK\n"), stdout);
      break;
    case NIS_PRIVATE_OBJ:
      fputs (_("PRIVATE\n"), stdout);
      break;
    default:
      fputs (_("(Unknown object)\n"), stdout);
      break;
    }
}

void
nis_print_rights (const unsigned int access)
{
  char result[17];
  unsigned int acc;
  int i;

  acc = access;			/* Parameter is const ! */
  result[i = 16] = '\0';
  while (i > 0)
    {
      i -= 4;
      result[i + 0] = (acc & NIS_READ_ACC) ? 'r' : '-';
      result[i + 1] = (acc & NIS_MODIFY_ACC) ? 'm' : '-';
      result[i + 2] = (acc & NIS_CREATE_ACC) ? 'c' : '-';
      result[i + 3] = (acc & NIS_DESTROY_ACC) ? 'd' : '-';

      acc >>= 8;
    }
  fputs (result, stdout);
}
libnsl_hidden_nolink_def (nis_print_rights, GLIBC_2_1)

void
nis_print_directory (const directory_obj *dir)
{
  nis_server *sptr;
  unsigned int i;

  printf (_("Name : `%s'\n"), dir->do_name);
  printf (_("Type : %s\n"), nis_nstype2str (dir->do_type));
  sptr = dir->do_servers.do_servers_val;
  for (i = 0; i < dir->do_servers.do_servers_len; i++)
    {
      if (i == 0)
	fputs (_("Master Server :\n"), stdout);
      else
	fputs (_("Replicate :\n"), stdout);
      printf (_("\tName       : %s\n"), sptr->name);
      fputs (_("\tPublic Key : "), stdout);
      switch (sptr->key_type)
	{
	case NIS_PK_NONE:
	  fputs (_("None.\n"), stdout);
	  break;
	case NIS_PK_DH:
	  printf (_("Diffie-Hellmann (%d bits)\n"),
		  (sptr->pkey.n_len - 1) * 4);
	  /* sptr->pkey.n_len counts the last 0, too */
	  break;
	case NIS_PK_RSA:
	  printf (_("RSA (%d bits)\n"), (sptr->pkey.n_len - 1) * 4);
	  break;
	case NIS_PK_KERB:
	  fputs (_("Kerberos.\n"), stdout);
	  break;
	default:
	  printf (_("Unknown (type = %d, bits = %d)\n"), sptr->key_type,
		  (sptr->pkey.n_len - 1) * 4);
	  break;
	}

      if (sptr->ep.ep_len != 0)
	{
	  unsigned int j;

	  endpoint *ptr;
	  ptr = sptr->ep.ep_val;
	  printf (_("\tUniversal addresses (%u)\n"), sptr->ep.ep_len);
	  for (j = 0; j < sptr->ep.ep_len; j++)
	    {
	      printf ("\t[%d] - ", j + 1);
	      if (ptr->proto != NULL && ptr->proto[0] != '\0')
		printf ("%s, ", ptr->proto);
	      else
		printf ("-, ");
	      if (ptr->family != NULL && ptr->family[0] != '\0')
		printf ("%s, ", ptr->family);
	      else
		printf ("-, ");
	      if (ptr->uaddr != NULL && ptr->uaddr[0] != '\0')
		printf ("%s\n", ptr->uaddr);
	      else
		fputs ("-\n", stdout);
	      ptr++;
	    }
	}
      sptr++;
    }

  fputs (_("Time to live : "), stdout);
  print_ttl (dir->do_ttl);
  fputs (_("Default Access rights :\n"), stdout);
  if (dir->do_armask.do_armask_len != 0)
    {
      oar_mask *ptr;

      ptr = dir->do_armask.do_armask_val;
      for (i = 0; i < dir->do_armask.do_armask_len; i++)
	{
	  nis_print_rights (ptr->oa_rights);
	  printf (_("\tType         : %s\n"),
	          nis_nstype2str ((nstype)ptr->oa_otype));
	  fputs (_("\tAccess rights: "), stdout);
	  nis_print_rights (ptr->oa_rights);
	  fputs ("\n", stdout);
	  ptr++;
	}
    }
}
libnsl_hidden_nolink_def (nis_print_directory, GLIBC_2_1)

void
nis_print_group (const group_obj *obj)
{
  unsigned int i;

  fputs (_("Group Flags :"), stdout);
  if (obj->gr_flags)
    printf ("0x%08X", obj->gr_flags);
  fputs (_("\nGroup Members :\n"), stdout);

  for (i = 0; i < obj->gr_members.gr_members_len; i++)
    printf ("\t%s\n", obj->gr_members.gr_members_val[i]);
}
libnsl_hidden_nolink_def (nis_print_group, GLIBC_2_1)

void
nis_print_table (const table_obj *obj)
{
  unsigned int i;

  printf (_("Table Type          : %s\n"), obj->ta_type);
  printf (_("Number of Columns   : %d\n"), obj->ta_maxcol);
  printf (_("Character Separator : %c\n"), obj->ta_sep);
  printf (_("Search Path         : %s\n"), obj->ta_path);
  fputs (_("Columns             :\n"), stdout);
  for (i = 0; i < obj->ta_cols.ta_cols_len; i++)
    {
      printf (_("\t[%d]\tName          : %s\n"), i,
	      obj->ta_cols.ta_cols_val[i].tc_name);
      fputs (_("\t\tAttributes    : "), stdout);
      print_flags (obj->ta_cols.ta_cols_val[i].tc_flags);
      fputs (_("\t\tAccess Rights : "), stdout);
      nis_print_rights (obj->ta_cols.ta_cols_val[i].tc_rights);
      fputc ('\n', stdout);
    }
}
libnsl_hidden_nolink_def (nis_print_table, GLIBC_2_1)

void
nis_print_link (const link_obj *obj)
{
  fputs (_("Linked Object Type : "), stdout);
  nis_print_objtype (obj->li_rtype);
  printf (_("Linked to : %s\n"), obj->li_name);
  /* XXX Print the attributes here, if they exists */
}
libnsl_hidden_nolink_def (nis_print_link, GLIBC_2_1)

void
nis_print_entry (const entry_obj *obj)
{
  unsigned int i;

  printf (_("\tEntry data of type %s\n"), obj->en_type);
  for (i = 0; i < obj->en_cols.en_cols_len; i++)
    {
      printf (_("\t[%u] - [%u bytes] "), i,
	      obj->en_cols.en_cols_val[i].ec_value.ec_value_len);
      if ((obj->en_cols.en_cols_val[i].ec_flags & EN_CRYPT) == EN_CRYPT)
	fputs (_("Encrypted data\n"), stdout);
      else if ((obj->en_cols.en_cols_val[i].ec_flags & EN_BINARY) == EN_BINARY)
	fputs (_("Binary data\n"), stdout);
      else if (obj->en_cols.en_cols_val[i].ec_value.ec_value_len == 0)
	fputs ("'(nil)'\n", stdout);
      else
	printf ("'%.*s'\n",
		(int)obj->en_cols.en_cols_val[i].ec_value.ec_value_len,
		obj->en_cols.en_cols_val[i].ec_value.ec_value_val);
    }
}
libnsl_hidden_nolink_def (nis_print_entry, GLIBC_2_1)

void
nis_print_object (const nis_object * obj)
{
  time_t buf;

  printf (_("Object Name   : %s\n"), obj->zo_name);
  printf (_("Directory     : %s\n"), obj->zo_domain);
  printf (_("Owner         : %s\n"), obj->zo_owner);
  printf (_("Group         : %s\n"), obj->zo_group);
  fputs (_("Access Rights : "), stdout);
  nis_print_rights (obj->zo_access);
  printf (_("\nTime to Live  : "));
  print_ttl (obj->zo_ttl);
  buf = obj->zo_oid.ctime;
  printf (_("Creation Time : %s"), ctime (&buf));
  buf = obj->zo_oid.mtime;
  printf (_("Mod. Time     : %s"), ctime (&buf));
  fputs (_("Object Type   : "), stdout);
  nis_print_objtype (obj->zo_data.zo_type);
  switch (obj->zo_data.zo_type)
    {
    case NIS_DIRECTORY_OBJ:
      nis_print_directory (&obj->zo_data.objdata_u.di_data);
      break;
    case NIS_GROUP_OBJ:
      nis_print_group (&obj->zo_data.objdata_u.gr_data);
      break;
    case NIS_TABLE_OBJ:
      nis_print_table (&obj->zo_data.objdata_u.ta_data);
      break;
    case NIS_ENTRY_OBJ:
      nis_print_entry (&obj->zo_data.objdata_u.en_data);
      break;
    case NIS_LINK_OBJ:
      nis_print_link (&obj->zo_data.objdata_u.li_data);
      break;
    case NIS_PRIVATE_OBJ:
      printf (_("    Data Length = %u\n"),
	      obj->zo_data.objdata_u.po_data.po_data_len);
      break;
    default:
      break;
    }
}
libnsl_hidden_nolink_def (nis_print_object, GLIBC_2_1)

void
nis_print_result (const nis_result *res)
{
  unsigned int i;

  printf (_("Status            : %s\n"), nis_sperrno (NIS_RES_STATUS (res)));
  printf (_("Number of objects : %u\n"), res->objects.objects_len);

  for (i = 0; i < res->objects.objects_len; i++)
    {
      printf (_("Object #%d:\n"), i);
      nis_print_object (&res->objects.objects_val[i]);
    }
}
libnsl_hidden_nolink_def (nis_print_result, GLIBC_2_1)
