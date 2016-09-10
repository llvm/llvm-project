using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LLVM.ClangTidy
{
    class MagicInheritance
    {
        public static readonly string Value = "{3A27184D-1774-489B-9BB7-7191B8E8E622}";
        public static readonly string Text = "<Inherit from project or parent>";
    }


    class DynamicPropertyConverter<T> : TypeConverter
    {
        private DynamicPropertyDescriptor<T> Descriptor_;
        private TypeConverter Root_;

        public DynamicPropertyConverter(DynamicPropertyDescriptor<T> Descriptor, TypeConverter Root)
        {
            Descriptor_ = Descriptor;
            Root_ = Root;
        }

        /// <summary>
        /// Returns true if there are specific values that can be chosen from a dropdown
        /// for this property.  Regardless of whether standard values are supported for
        /// the underlying type, we always support standard values because we need to
        /// display the inheritance option.
        /// </summary>
        /// <returns>true</returns>
        public override bool GetStandardValuesSupported(ITypeDescriptorContext context)
        {
            return true;
        }

        /// <summary>
        /// Get the set of all standard values that can be chosen from a dropdown for this
        /// property.  If the underlying type supports standard values, we want to include
        /// all those.  Additionally, we want to display the option to inherit the value,
        /// but only if the value is not already inheriting.
        /// </summary>
        public override StandardValuesCollection GetStandardValues(ITypeDescriptorContext context)
        {
            List<object> Values = new List<object>();
            if (Root_.GetStandardValuesSupported(context))
            {
                StandardValuesCollection RootValues = Root_.GetStandardValues(context);
                Values.AddRange(RootValues.Cast<object>());
            }
            if (!Descriptor_.IsInheriting)
                Values.Add(MagicInheritance.Value);
            StandardValuesCollection Result = new StandardValuesCollection(Values);
            return Result;
        }

        /// <summary>
        /// Determines whether this property can accept values other than those specified
        /// in the dropdown (for example by manually typing into the field).
        /// </summary>
        public override bool GetStandardValuesExclusive(ITypeDescriptorContext context)
        {
            // Although we add items to the dropdown list, we do not change whether or not
            // the set of values are exclusive.  If the user could type into the field before
            // they still can.  And if they couldn't before, they still can't.
            return Root_.GetStandardValuesExclusive(context);
        }

        public override bool CanConvertFrom(ITypeDescriptorContext context, Type sourceType)
        {
            return Root_.CanConvertFrom(context, sourceType);
        }

        public override bool CanConvertTo(ITypeDescriptorContext context, Type destinationType)
        {
            return Root_.CanConvertTo(context, destinationType);
        }

        public override object ConvertFrom(ITypeDescriptorContext context, CultureInfo culture, object value)
        {
            if (value.Equals(MagicInheritance.Value))
                return MagicInheritance.Text;
            return Root_.ConvertFrom(context, culture, value);
        }

        public override object ConvertTo(ITypeDescriptorContext context, CultureInfo culture, object value, Type destinationType)
        {
            if (value.GetType() == destinationType)
                return value;

            return Root_.ConvertTo(context, culture, value, destinationType);
        }

        public override object CreateInstance(ITypeDescriptorContext context, IDictionary propertyValues)
        {
            return Root_.CreateInstance(context, propertyValues);
        }

        public override bool Equals(object obj)
        {
            return Root_.Equals(obj);
        }

        public override bool GetCreateInstanceSupported(ITypeDescriptorContext context)
        {
            return Root_.GetCreateInstanceSupported(context);
        }

        public override int GetHashCode()
        {
            return Root_.GetHashCode();
        }

        public override PropertyDescriptorCollection GetProperties(ITypeDescriptorContext context, object value, Attribute[] attributes)
        {
            return Root_.GetProperties(context, value, attributes);
        }

        public override bool GetPropertiesSupported(ITypeDescriptorContext context)
        {
            return Root_.GetPropertiesSupported(context);
        }

        public override bool IsValid(ITypeDescriptorContext context, object value)
        {
            return Root_.IsValid(context, value);
        }

        public override string ToString()
        {
            return Root_.ToString();
        }
    }
}
